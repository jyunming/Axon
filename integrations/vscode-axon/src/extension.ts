import * as vscode from 'vscode';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { spawn, ChildProcess } from 'child_process';

let serverProcess: ChildProcess | undefined;
let externalServerPid: number | undefined; // PID of a server we didn't spawn but own on deactivate
let outputChannel: vscode.OutputChannel;

// ---------------------------------------------------------------------------
// Graph Panel
// ---------------------------------------------------------------------------

class AxonGraphPanel {
  static currentPanel: AxonGraphPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposables: vscode.Disposable[] = [];

  static createOrReveal(context: vscode.ExtensionContext): AxonGraphPanel {
    if (AxonGraphPanel.currentPanel) {
      AxonGraphPanel.currentPanel._panel.reveal(vscode.ViewColumn.Beside);
      return AxonGraphPanel.currentPanel;
    }
    const mediaUri = vscode.Uri.joinPath(context.extensionUri, 'media');
    const panel = vscode.window.createWebviewPanel(
      'axonGraph', 'Axon Graph', vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [mediaUri],
      }
    );
    AxonGraphPanel.currentPanel = new AxonGraphPanel(panel, context.extensionUri);
    return AxonGraphPanel.currentPanel;
  }

  private _disposed = false;

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._extensionUri = extensionUri;
    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
    this._panel.webview.onDidReceiveMessage(
      (msg: any) => this._handleMessage(msg),
      null, this._disposables
    );
    this._panel.webview.html = this._loadingHtml('Axon Graph');
  }

  update(data: { query: string; answer: string; sources: any[]; knowledgeGraph: any; codeGraph: any }) {
    this._panel.title = `Axon: ${data.query.slice(0, 40)}`;
    this._panel.webview.html = this._buildHtml(data);
  }

  showLoading(query: string) {
    this._panel.title = `Axon: ${query.slice(0, 40)}…`;
    this._panel.webview.html = this._loadingHtml(query);
  }

  private _handleMessage(msg: any) {
    if (msg.command === 'openFile') {
      const uri = vscode.Uri.file(msg.path);
      vscode.window.showTextDocument(uri, {
        selection: new vscode.Range(
          Math.max(0, (msg.line || 1) - 1), 0,
          Math.max(0, (msg.line || 1) - 1), 0
        )
      });
    }
  }

  private _loadingHtml(query: string): string {
    const csp = `default-src 'none'; style-src 'unsafe-inline';`;
    return `<!DOCTYPE html><html><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="${csp}"></head><body style="background:#1e1e1e;color:#ccc;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
      <div style="text-align:center">
        <div style="font-size:2em;margin-bottom:1em">⟳</div>
        <div>Loading graph for: <em>${escapeHtml(query)}</em></div>
      </div>
    </body></html>`;
  }

  private _buildHtml(data: { query: string; answer: string; sources: any[]; knowledgeGraph: any; codeGraph: any }): string {
    // Serialize data for the webview — injected via <script type="application/json">,
    // which the browser treats as opaque data (never executed), so it is NOT subject
    // to script-src CSP. This eliminates the need for 'unsafe-inline'.
    const dataJson = JSON.stringify(data).replace(/<\/script>/gi, '<\\/script>');

    const forceGraphUri = this._panel.webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', '3d-force-graph.min.js')
    );
    const panelJsUri = this._panel.webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'graph-panel.js')
    );

    // No inline scripts → no 'unsafe-inline' needed.  Just cspSource covers both external files.
    const cspSrc = this._panel.webview.cspSource;
    const csp = `default-src 'none'; script-src ${cspSrc}; style-src 'unsafe-inline';`;

    return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1e1e1e; color: #d4d4d4; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; height: 100vh; display: flex; overflow: hidden; }
  #left { width: 35%; min-width: 260px; display: flex; flex-direction: column; border-right: 1px solid #333; overflow: hidden; }
  #right { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #tab-bar { display: flex; border-bottom: 1px solid #333; background: #252526; flex-shrink: 0; }
  .tab { padding: 6px 16px; font-size: 0.78em; cursor: pointer; border-bottom: 2px solid transparent; color: #888; user-select: none; }
  .tab:hover { color: #d4d4d4; }
  .tab-active { color: #d4d4d4; border-bottom-color: #569cd6; }
  .tab-disabled { opacity: 0.38; cursor: not-allowed; }
  #graph-area { flex: 1; position: relative; overflow: hidden; }
  #graph-kg, #graph-cg { position: absolute; inset: 0; display: none; }
  #graph-placeholder { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; color: #888; font-style: italic; padding: 24px; text-align: center; }
  .graph-tooltip { position: absolute; display: none; background: #252526; color: #d4d4d4; padding: 6px 10px; border-radius: 4px; font-size: 0.78em; max-width: 260px; pointer-events: none; border: 1px solid #444; z-index: 10; word-break: break-word; }
  #query-text { padding: 12px 16px; font-size: 0.85em; color: #9cdcfe; border-bottom: 1px solid #333; font-weight: 600; word-break: break-word; }
  #answer-text { padding: 12px 16px; font-size: 0.82em; line-height: 1.5; overflow-y: auto; flex: 1; border-bottom: 1px solid #333; }
  #citations { overflow-y: auto; max-height: 200px; padding: 8px; }
  #citations-heading { padding: 6px 16px; font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.05em; background: #252526; border-bottom: 1px solid #333; }
  .citation { padding: 6px 10px; cursor: pointer; border-bottom: 1px solid #2a2a2a; font-size: 0.78em; }
  .citation:hover { background: #2a2d2e; }
  .cite-num { color: #569cd6; margin-right: 4px; }
  .cite-src { color: #4ec9b0; font-weight: 500; }
  .cite-text { color: #888; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
</style>
</head>
<body>
<div id="left">
  <div id="query-text"></div>
  <div id="answer-text"></div>
  <div id="citations-heading">Sources</div>
  <div id="citations"></div>
</div>
<div id="right">
  <div id="tab-bar">
    <div id="tab-kg" class="tab">Knowledge Graph</div>
    <div id="tab-cg" class="tab">Code Graph</div>
  </div>
  <div id="graph-area">
    <div id="graph-kg"></div>
    <div id="graph-cg"></div>
    <div id="graph-placeholder">No graph data available.<br>Ingest with <code>graph_rag: true</code> or <code>code_graph: true</code>.</div>
  </div>
</div>
<!-- Data passed as non-executable JSON — exempt from script-src CSP -->
<script type="application/json" id="app-data">${dataJson}</script>
<!-- Two external scripts only — no inline JS, satisfies strict script-src CSP -->
<script src="${forceGraphUri}"></script>
<script src="${panelJsUri}"></script>
</body>
</html>`;
  }

  dispose() {
    if (this._disposed) { return; }
    this._disposed = true;
    AxonGraphPanel.currentPanel = undefined;
    this._panel.dispose();
    this._disposables.forEach(d => d.dispose());
    this._disposables = [];
  }
}

function escapeHtml(str: string): string {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function showGraphForQuery(
  context: vscode.ExtensionContext,
  query: string
): Promise<'opened' | 'updated' | 'no_graph_available' | 'query_failed'> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const apiKey = config.get<string>('apiKey', '');

  const panel = AxonGraphPanel.createOrReveal(context);
  panel.showLoading(query);

  try {
    const [queryRes, searchRes, kgRes, cgRes] = await Promise.all([
      httpPost(`${apiBase}/query`, { query, discuss: false }, apiKey),
      httpPost(`${apiBase}/search/raw`, { query }, apiKey),
      httpGet(`${apiBase}/graph/data`, apiKey),
      httpGet(`${apiBase}/code-graph/data`, apiKey),
    ]);

    const answer = JSON.parse(queryRes.body);
    const search = JSON.parse(searchRes.body);
    const knowledgeGraph = JSON.parse(kgRes.body);
    const codeGraph      = JSON.parse(cgRes.body);

    if (queryRes.status !== 200) {
      vscode.window.showErrorMessage(`Axon query failed: ${formatDetail(answer, queryRes.body)}`);
      panel.dispose();
      return 'query_failed';
    }

    panel.update({
      query,
      answer: answer.response || '',
      sources: search.results ?? [],
      knowledgeGraph,
      codeGraph,
    });

    const hasAny = (knowledgeGraph.nodes?.length ?? 0) > 0 || (codeGraph.nodes?.length ?? 0) > 0;
    return hasAny ? 'opened' : 'no_graph_available';
  } catch (err) {
    vscode.window.showErrorMessage(`Axon graph panel error: ${err}`);
    panel.dispose();
    return 'query_failed';
  }
}

async function showGraphForSelection(context: vscode.ExtensionContext) {
  const editor = vscode.window.activeTextEditor;
  const query = editor?.document.getText(editor.selection).trim()
             || await vscode.window.showInputBox({ prompt: 'Enter query for Axon graph' });
  if (!query) { return; }
  await showGraphForQuery(context, query);
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  outputChannel = vscode.window.createOutputChannel('Axon');
  context.subscriptions.push(outputChannel);

  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const autoStart = config.get<boolean>('autoStart', true);

  outputChannel.appendLine(`Axon extension activating. API base: ${apiBase}`);

  if (autoStart) {
    await ensureServerRunning(apiBase, context);
  }

  const useCopilotLlm = config.get<boolean>('useCopilotLlm', false);
  const apiKey = config.get<string>('apiKey', '');
  if (useCopilotLlm) {
    outputChannel.appendLine('Axon: Using Copilot LLM for backend tasks.');
    // Start the worker loop
    startCopilotLlmWorker(apiBase, apiKey);
    // Tell the backend to use the 'copilot' provider and PERSIST it
    waitForHealth(apiBase, 120_000).then((running) => {
      if (running) {
        httpPost(`${apiBase}/config/update`, { llm_provider: 'copilot', persist: true }, apiKey)
          .then(() => outputChannel.appendLine('Axon backend configured to use Copilot provider (persistent).'))
          .catch((err) => outputChannel.appendLine(`Failed to set copilot provider: ${err}`));
      }
    });
  }

  // Register the @axon chat participant
  const participant = vscode.chat.createChatParticipant('axon.chat', chatHandler);
  participant.iconPath = new vscode.ThemeIcon('database');
  context.subscriptions.push(participant);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('axon.switchProject', () => switchProject(apiBase)),
    vscode.commands.registerCommand('axon.createProject', () => createNewProject(apiBase)),
    vscode.commands.registerCommand('axon.ingestFile', () => ingestCurrentFile(apiBase)),
    vscode.commands.registerCommand('axon.ingestWorkspace', () => ingestWorkspaceFolder(apiBase)),
    vscode.commands.registerCommand('axon.ingestFolder', () => ingestPickedFolder(apiBase)),
    vscode.commands.registerCommand('axon.startServer', () => ensureServerRunning(apiBase, context)),
    vscode.commands.registerCommand('axon.stopServer', () => stopServer()),
    vscode.commands.registerCommand('axon.initStore', () => initStore(apiBase)),
    vscode.commands.registerCommand('axon.shareProject', () => shareProject(apiBase)),
    vscode.commands.registerCommand('axon.redeemShare', () => redeemShare(apiBase)),
    vscode.commands.registerCommand('axon.revokeShare', () => revokeShare(apiBase)),
    vscode.commands.registerCommand('axon.listShares', () => listShares(apiBase)),
    vscode.commands.registerCommand('axon.refreshIngest', () => refreshIngest(apiBase)),
    vscode.commands.registerCommand('axon.listStaleDocs', () => listStaleDocs(apiBase)),
    vscode.commands.registerCommand('axon.clearKnowledgeBase', () => clearKnowledgeBase(apiBase)),
    vscode.commands.registerCommand('axon.showGraphStatus', () => showGraphStatus(apiBase)),
    vscode.commands.registerCommand('axon.showGraphForQuery', async () => {
      const query = await vscode.window.showInputBox({ prompt: 'Axon: Enter query to visualise' });
      if (query) { await showGraphForQuery(context, query); }
    }),
    vscode.commands.registerCommand('axon.showGraphForSelection', () => showGraphForSelection(context)),
  );

  // Register Language Model Tools (for Copilot Agent toolset)
  try {
    if ('lm' in vscode && (vscode as any).lm.registerTool) {
      outputChannel.appendLine('Registering Axon Language Model Tools...');
      context.subscriptions.push(
        (vscode as any).lm.registerTool('axon_searchKnowledge', new AxonSearchTool()),
        (vscode as any).lm.registerTool('axon_queryKnowledge', new AxonQueryTool()),
        (vscode as any).lm.registerTool('axon_ingestText', new AxonIngestTextTool()),
        (vscode as any).lm.registerTool('axon_ingestUrl', new AxonIngestUrlTool()),
        (vscode as any).lm.registerTool('axon_ingestPath', new AxonIngestPathTool()),
        (vscode as any).lm.registerTool('axon_getIngestStatus', new AxonGetIngestStatusTool()),
        (vscode as any).lm.registerTool('axon_listProjects', new AxonListProjectsTool()),
        (vscode as any).lm.registerTool('axon_switchProject', new AxonSwitchProjectTool()),
        (vscode as any).lm.registerTool('axon_createProject', new AxonCreateProjectTool()),
        (vscode as any).lm.registerTool('axon_deleteProject', new AxonDeleteProjectTool()),
        (vscode as any).lm.registerTool('axon_deleteDocuments', new AxonDeleteDocumentsTool()),
        (vscode as any).lm.registerTool('axon_getCollection', new AxonGetCollectionTool()),
        (vscode as any).lm.registerTool('axon_clearCollection', new AxonClearCollectionTool()),
        (vscode as any).lm.registerTool('axon_updateSettings', new AxonUpdateSettingsTool()),
        (vscode as any).lm.registerTool('axon_listShares', new AxonListSharesTool()),
        (vscode as any).lm.registerTool('axon_initStore', new AxonInitStoreTool()),
        (vscode as any).lm.registerTool('axon_ingestImage', new AxonIngestImageTool()),
        (vscode as any).lm.registerTool('axon_showGraph', new AxonShowGraphTool(context))
      );
      outputChannel.appendLine('Successfully registered all Axon tools.');
    } else {
      outputChannel.appendLine('Language Model Tools API not available in this VS Code version.');
    }
  } catch (err) {
    outputChannel.appendLine(`Error registering tools: ${err}`);
  }

  outputChannel.appendLine('Axon extension ready.');
}

export function deactivate(): void {
  stopServer();
}

// ---------------------------------------------------------------------------
// Language Model Tools (Copilot Agent Toolset)
// ---------------------------------------------------------------------------

class AxonSearchTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Searching Axon knowledge base for: "${options.input.query}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { query, topK = 5, threshold } = options.input;

    try {
      const chunks = await searchAxon(apiBase, apiKey, query, topK, threshold);
      const content = chunks.map(c => `[ID: ${c.id}] Source: ${c.metadata?.source}\n${c.text}`).join('\n\n---\n\n');
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(content || 'No results found.')]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during Axon search: ${err}`)]);
    }
  }
}

class AxonQueryTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Querying Axon knowledge base: "${options.input.query}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { query, top_k } = options.input;

    try {
      // discuss: false disables the general-knowledge fallback so the tool
      // returns a definitive "no data" message instead of a hallucinated answer
      // when the knowledge base has no matching content.
      const body: any = { query, discuss: false };
      if (top_k != null) { body.top_k = top_k; }
      const result = await httpPost(`${apiBase}/query`, body, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(data.response || 'No answer generated.')]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error calling Axon query: ${err}`)]);
    }
  }
}

class AxonIngestTextTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { text, source = 'agent_input' } = options.input;

    try {
      const result = await httpPost(`${apiBase}/add_text`, { text, metadata: { source } }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Success: ${data.status}, ID: ${data.doc_id}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during text ingest: ${err}`)]);
    }
  }
}

class AxonIngestUrlTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { url } = options.input;

    try {
      const result = await httpPost(`${apiBase}/ingest_url`, { url }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon URL Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, URL: ${data.url}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during URL ingest: ${err}`)]);
    }
  }
}

class AxonIngestPathTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Ingesting local path into Axon: "${options.input.path}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { path } = options.input;

    try {
      const result = await httpPost(`${apiBase}/ingest`, { path }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Path Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Message: ${data.message}, JobID: ${data.job_id}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during path ingest: ${err}`)]);
    }
  }
}

class AxonGetIngestStatusTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { job_id } = options.input;

    try {
      const result = await httpGet(`${apiBase}/ingest/status/${encodeURIComponent(job_id)}`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status === 404) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Job not found: ${job_id}`)]);
      }
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error checking status (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      const status = data.status;
      if (status === 'completed') {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Ingestion complete. Status: completed. You can now search the ingested documents.`)]);
      } else if (status === 'failed') {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Ingestion failed. Error: ${data.error || 'unknown error'}`)]);
      } else {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Ingestion still in progress (status: ${status}). Wait a moment and check again.`)]);
      }
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error checking ingest status: ${err}`)]);
    }
  }
}

class AxonListProjectsTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');

    try {
      const result = await httpGet(`${apiBase}/projects`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      const names = (data.projects || []).map((p: any) => p.name).join(', ');
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Projects: ${names || 'None'}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error listing projects: ${err}`)]);
    }
  }
}

class AxonSwitchProjectTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Switching Axon active project to: "${options.input.name}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { name } = options.input;

    try {
      const result = await httpPost(`${apiBase}/project/switch`, { name }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Switched to project: ${data.active_project || name}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error switching project: ${err}`)]);
    }
  }
}

class AxonCreateProjectTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Creating Axon project: "${options.input.name}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { name, description = '' } = options.input;

    try {
      const result = await httpPost(`${apiBase}/project/new`, { name, description }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Project Creation Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Project: ${data.project}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error creating project: ${err}`)]);
    }
  }
}

class AxonDeleteProjectTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Deleting Axon project: "${options.input.name}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { name } = options.input;

    try {
      const result = await httpPost(`${apiBase}/project/delete/${name}`, {}, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Project Deletion Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Message: ${data.message}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error deleting project: ${err}`)]);
    }
  }
}

class AxonDeleteDocumentsTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Deleting ${options.input.docIds.length} documents from Axon...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { docIds } = options.input;

    try {
      const result = await httpPost(`${apiBase}/delete`, { doc_ids: docIds }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Document Deletion Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Deleted: ${data.deleted}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error deleting documents: ${err}`)]);
    }
  }
}

class AxonGetCollectionTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');

    try {
      const result = await httpGet(`${apiBase}/collection`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      const files = (data.files || []).map((f: any) => `${f.source} (${f.chunks} chunks)`).join('\n');
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Total Files: ${data.total_files}\nTotal Chunks: ${data.total_chunks}\n\nFiles:\n${files}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error getting collection status: ${err}`)]);
    }
  }
}

class AxonClearCollectionTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Clearing all data from the active Axon project...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');

    try {
      const result = await httpPost(`${apiBase}/clear`, {}, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Clear Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Message: ${data.message}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error clearing collection: ${err}`)]);
    }
  }
}

class AxonUpdateSettingsTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, token: vscode.CancellationToken) {
    return {
      invocationMessage: `Updating Axon RAG settings...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');

    try {
      const result = await httpPost(`${apiBase}/config/update`, options.input, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Configuration Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Settings Applied.`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error updating settings: ${err}`)]);
    }
  }
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

async function getPortPid(port: number): Promise<number | undefined> {
  try {
    const { execSync } = require('child_process');
    if (process.platform === 'win32') {
      const out = execSync(`powershell -Command "Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"`, { encoding: 'utf8' }).trim();
      const pid = parseInt(out, 10);
      return isNaN(pid) ? undefined : pid;
    } else {
      const out = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf8' }).trim();
      const pid = parseInt(out, 10);
      return isNaN(pid) ? undefined : pid;
    }
  } catch {
    return undefined;
  }
}

/**
 * Discover the Python interpreter that has Axon installed.
 *
 * Probe order:
 *   1. axon.pythonPath VS Code setting (explicit user override)
 *   2. ~/.axon/.python_path written by the `axon` CLI on first run (pip / venv / pipx)
 *   3. pipx isolated venv (predictable fixed path for `pipx install axon`)
 *   4. .venv / venv / env inside the open workspace folder
 *   5. System Python (python3 → python)
 *
 * Shows a one-time notification if nothing is found with axon importable.
 */
async function discoverPythonPath(): Promise<string> {
  const config = vscode.workspace.getConfiguration('axon');
  const isWin = process.platform === 'win32';

  // 1. Explicit user setting
  const explicit = config.get<string>('pythonPath', '');
  if (explicit) {
    return explicit;
  }

  // 2. ~/.axon/.python_path written by `axon` CLI on first run
  const discoveryFile = path.join(os.homedir(), '.axon', '.python_path');
  if (fs.existsSync(discoveryFile)) {
    const discovered = fs.readFileSync(discoveryFile, 'utf8').trim();
    if (discovered && fs.existsSync(discovered)) {
      outputChannel.appendLine(`Python auto-detected via ~/.axon/.python_path: ${discovered}`);
      return discovered;
    }
  }

  // 3. pipx isolated venv (fixed path for `pipx install axon`)
  const pipxPython = isWin
    ? path.join(os.homedir(), '.local', 'pipx', 'venvs', 'axon', 'Scripts', 'python.exe')
    : path.join(os.homedir(), '.local', 'pipx', 'venvs', 'axon', 'bin', 'python');
  if (fs.existsSync(pipxPython)) {
    outputChannel.appendLine(`Python auto-detected via pipx venv: ${pipxPython}`);
    return pipxPython;
  }

  // 4. Workspace venv (.venv, venv, env)
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (workspaceFolders) {
    for (const folder of workspaceFolders) {
      for (const venvDir of ['.venv', 'venv', 'env']) {
        const candidate = isWin
          ? path.join(folder.uri.fsPath, venvDir, 'Scripts', 'python.exe')
          : path.join(folder.uri.fsPath, venvDir, 'bin', 'python');
        if (fs.existsSync(candidate)) {
          outputChannel.appendLine(`Python auto-detected via workspace venv (${venvDir}): ${candidate}`);
          return candidate;
        }
      }
    }
  }

  // 5. System Python fallback
  const systemPython = isWin ? 'python' : 'python3';
  outputChannel.appendLine(
    `Python auto-detection: no venv found. Falling back to system ${systemPython}. ` +
    `If Axon is not found, run \`axon\` once after installation, or set axon.pythonPath.`
  );

  // Show a one-time notification so users know what to do
  const msg = 'Axon: Python auto-detection did not find an Axon installation. ' +
    'Run `axon` once after installing, or set the `axon.pythonPath` setting.';
  vscode.window.showWarningMessage(msg, 'Open Settings').then(choice => {
    if (choice === 'Open Settings') {
      vscode.commands.executeCommand('workbench.action.openSettings', 'axon.pythonPath');
    }
  });

  return systemPython;
}

async function ensureServerRunning(apiBase: string, context: vscode.ExtensionContext): Promise<void> {
  if (await isAxonRunning(apiBase)) {
    outputChannel.appendLine('Axon API already running.');
    // Capture PID so we can stop it on deactivate even if we didn't spawn it
    const portMatch = apiBase.match(/:(\d+)/);
    const port = portMatch ? parseInt(portMatch[1], 10) : 8000;
    externalServerPid = await getPortPid(port);
    if (externalServerPid) {
      outputChannel.appendLine(`Tracking external server PID: ${externalServerPid}`);
    }
    return;
  }

  const config = vscode.workspace.getConfiguration('axon');
  const pythonPath = await discoverPythonPath();

  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    outputChannel.appendLine('No workspace folder open. Cannot auto-start Axon server.');
    return;
  }
  const workspaceRoot = workspaceFolders[0].uri.fsPath;

  // Parse port from apiBase (e.g. http://localhost:8000 -> 8000)
  let port = "8000";
  try {
    const url = new URL(apiBase);
    if (url.port) {
      port = url.port;
    }
  } catch (err) {
    outputChannel.appendLine(`Warning: Could not parse port from apiBase "${apiBase}". Defaulting to 8000.`);
  }

  // Allow ingesting any local path by setting RAG_INGEST_BASE to the filesystem root.
  // Users can override this via axon.ingestBase in settings.
  const configuredBase = config.get<string>('ingestBase', '');
  const fsRoot = configuredBase || (process.platform === 'win32'
    ? path.parse(workspaceRoot).root  // e.g. "C:\"
    : '/');

  const storeBase = config.get<string>('storeBase', '');

  outputChannel.appendLine(`Starting Axon API server with: ${pythonPath} -m uvicorn axon.api:app --host 127.0.0.1 --port ${port}`);
  outputChannel.appendLine(`RAG_INGEST_BASE=${fsRoot} (any path under this root is ingestable)`);
  if (storeBase) {
    outputChannel.appendLine(`AXON_STORE_BASE=${storeBase}`);
  }

  serverProcess = spawn(pythonPath, ['-m', 'uvicorn', 'axon.api:app', '--host', '127.0.0.1', '--port', port], {
    cwd: workspaceRoot,
    shell: true, // Crucial for Windows path resolution
    env: {
      ...process.env,
      PYTHONPATH: path.join(workspaceRoot, 'src'),
      RAG_INGEST_BASE: fsRoot,
      ...(storeBase ? { AXON_STORE_BASE: storeBase } : {}),
    },
  });

  serverProcess.stdout?.on('data', (data: Buffer) => {
    outputChannel.append(`[server] ${data.toString()}`);
  });
  serverProcess.stderr?.on('data', (data: Buffer) => {
    outputChannel.append(`[server] ${data.toString()}`);
  });
  serverProcess.on('exit', (code) => {
    outputChannel.appendLine(`Axon server exited with code ${code}`);
    serverProcess = undefined;
  });

  const started = await waitForHealth(apiBase, 120_000); // 2 min — model loading takes time

  if (started) {
    outputChannel.appendLine('Axon API server is ready.');
    vscode.window.showInformationMessage('Axon API server started successfully.');
  } else {
    outputChannel.appendLine('Axon API server did not become ready within 30 seconds.');
    vscode.window.showWarningMessage('Axon API server failed to start. Check the Axon output panel.');
  }
}

function stopServer(): void {
  if (serverProcess) {
    outputChannel.appendLine('Stopping Axon API server...');
    serverProcess.kill();
    serverProcess = undefined;
  } else if (externalServerPid) {
    outputChannel.appendLine(`Stopping external Axon API server (PID ${externalServerPid})...`);
    try {
      process.kill(externalServerPid);
    } catch {
      // Process may have already exited
    }
    externalServerPid = undefined;
  }
}

async function isAxonRunning(apiBase: string): Promise<boolean> {
  try {
    const result = await httpGet(`${apiBase}/health`);
    return result.status === 200;
  } catch {
    return false;
  }
}

async function waitForHealth(apiBase: string, timeoutMs: number): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await isAxonRunning(apiBase)) {
      return true;
    }
    await sleep(500);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Chat participant handler
// ---------------------------------------------------------------------------

async function chatHandler(
  request: vscode.ChatRequest,
  _context: vscode.ChatContext,
  response: vscode.ChatResponseStream,
  token: vscode.CancellationToken,
): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const apiKey = config.get<string>('apiKey', '');
  const topK = config.get<number>('topK', 5);

  // 1. Retrieve relevant chunks from Axon
  let contextText = '';
  try {
    const chunks = await searchAxon(apiBase, apiKey, request.prompt, topK);
    if (chunks.length > 0) {
      contextText = chunks
        .map((c, i) => `[${i + 1}] Source: ${c.metadata?.source ?? 'unknown'}\n${c.text}`)
        .join('\n\n---\n\n');
      outputChannel.appendLine(`Retrieved ${chunks.length} chunks for query: "${request.prompt}"`);
    } else {
      outputChannel.appendLine('No chunks found for query.');
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    response.markdown(`> **Axon error**: Could not reach the API at \`${apiBase}\`. ${msg}\n\nMake sure the Axon server is running (or enable \`axon.autoStart\`).`);
    return;
  }

  // 2. Select a Copilot language model
  const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
  if (models.length === 0) {
    response.markdown('> **No Copilot language model available.** Make sure GitHub Copilot is installed and signed in.');
    return;
  }
  const model = models[0];

  // 3. Build messages
  const systemContent = contextText
    ? `You are a helpful assistant. Use the following excerpts from the knowledge base to answer the user's question. If the excerpts do not contain sufficient information, say so.\n\n--- Knowledge Base Excerpts ---\n\n${contextText}\n\n--- End of Excerpts ---`
    : 'You are a helpful assistant. No relevant knowledge base content was found for this query.';

  const messages = [
    vscode.LanguageModelChatMessage.User(systemContent + '\n\n' + request.prompt),
  ];

  // 4. Stream the Copilot response
  try {
    const chatResponse = await model.sendRequest(messages, {}, token);
    for await (const chunk of chatResponse.text) {
      response.markdown(chunk);
    }
  } catch (err) {
    if (err instanceof vscode.LanguageModelError) {
      response.markdown(`> **Copilot error**: ${err.message} (${err.code})`);
    } else {
      throw err;
    }
  }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async function switchProject(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');

  let projects: any[] = [];
  try {
    const result = await httpGet(`${apiBase}/projects`, apiKey);
    const data = JSON.parse(result.body);
    projects = data.projects ?? [];
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to list projects. Is the server running?`);
    return;
  }

  if (projects.length === 0) {
    vscode.window.showInformationMessage('Axon: No projects found.');
    return;
  }

  const selected = await vscode.window.showQuickPick(projects.map(p => p.name), {
    placeHolder: 'Select an Axon project',
  });
  if (!selected) {
    return;
  }

  try {
    await httpPost(`${apiBase}/project/switch`, { name: selected }, apiKey);
    vscode.window.showInformationMessage(`Axon: Switched to project "${selected}".`);
    outputChannel.appendLine(`Switched to project: ${selected}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to switch project.`);
  }
}

async function createNewProject(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');

  const name = await vscode.window.showInputBox({
    prompt: 'Enter a name for the new Axon project',
    placeHolder: 'e.g. project-alpha'
  });
  if (!name) {
    return;
  }

  const description = await vscode.window.showInputBox({
    prompt: 'Optional: Enter a description',
    placeHolder: 'Documentation for system architecture...'
  });

  try {
    await httpPost(`${apiBase}/project/new`, { name, description: description || '' }, apiKey);
    vscode.window.showInformationMessage(`Axon: Created project "${name}".`);
    outputChannel.appendLine(`Created project: ${name}`);
    // Auto-switch to it
    await httpPost(`${apiBase}/project/switch`, { name }, apiKey);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to create project.`);
  }
}

async function ingestWorkspaceFolder(apiBase: string): Promise<void> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    vscode.window.showWarningMessage('Axon: No workspace folder is open.');
    return;
  }

  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const folderPath = workspaceFolders[0].uri.fsPath;

  try {
    const result = await httpPost(`${apiBase}/ingest`, { path: folderPath }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Ingest failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: Ingesting workspace folder "${folderPath}" (job ${data.job_id}). Check the Axon output panel for progress.`
    );
    outputChannel.appendLine(`Ingest workspace: ${folderPath} — job ${data.job_id}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to ingest workspace folder. Is the server running?`);
  }
}

async function ingestPickedFolder(apiBase: string): Promise<void> {
  const selected = await vscode.window.showOpenDialog({
    canSelectFolders: true,
    canSelectFiles: true,
    canSelectMany: false,
    openLabel: 'Ingest into Axon',
    title: 'Select a folder or file to ingest into Axon',
  });

  if (!selected || selected.length === 0) {
    return;
  }

  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const selectedPath = selected[0].fsPath;

  try {
    const result = await httpPost(`${apiBase}/ingest`, { path: selectedPath }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Ingest failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: Ingesting "${selectedPath}" (job ${data.job_id}). Check the Axon output panel for progress.`
    );
    outputChannel.appendLine(`Ingest folder/file: ${selectedPath} — job ${data.job_id}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to ingest. Is the server running?`);
  }
}

async function ingestCurrentFile(apiBase: string): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('Axon: No active editor.');
    return;
  }

  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');

  const filePath = editor.document.uri.fsPath;
  const text = editor.document.getText();
  const source = path.basename(filePath);

  try {
    await httpPost(`${apiBase}/add_texts`, {
      docs: [{ text, metadata: { source: filePath } }],
    }, apiKey);
    vscode.window.showInformationMessage(`Axon: Ingested "${source}" into the knowledge base.`);
    outputChannel.appendLine(`Ingested file: ${filePath}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to ingest file. Is the server running?`);
  }
}

// ---------------------------------------------------------------------------
// HTTP helpers (no external deps — built-in Node.js http/https)
// ---------------------------------------------------------------------------

interface HttpResult {
  status: number;
  body: string;
}

/** Safely serialize a FastAPI error detail, which may be a string, array, or object. */
function formatDetail(data: any, fallback: string): string {
  const d = data?.detail;
  if (d === undefined || d === null) { return fallback; }
  if (typeof d === 'string') { return d; }
  return JSON.stringify(d);
}

function httpGet(url: string, apiKey?: string): Promise<HttpResult> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const lib = parsed.protocol === 'https:' ? https : http;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }
    const req = lib.get({ hostname: parsed.hostname, port: parsed.port, path: parsed.pathname + parsed.search, headers }, (res) => {
      let body = '';
      res.on('data', (chunk: Buffer) => { body += chunk.toString(); });
      res.on('end', () => resolve({ status: res.statusCode ?? 0, body }));
    });
    req.on('error', reject);
    req.setTimeout(5000, () => { req.destroy(); reject(new Error('Request timed out')); });
  });
}

function httpPost(url: string, payload: unknown, apiKey?: string): Promise<HttpResult> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const lib = parsed.protocol === 'https:' ? https : http;
    const body = JSON.stringify(payload);
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body).toString(),
    };
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }
    const req = lib.request(
      { method: 'POST', hostname: parsed.hostname, port: parsed.port, path: parsed.pathname, headers },
      (res) => {
        let resBody = '';
        res.on('data', (chunk: Buffer) => { resBody += chunk.toString(); });
        res.on('end', () => resolve({ status: res.statusCode ?? 0, body: resBody }));
      },
    );
    req.on('error', reject);
    req.setTimeout(20_000, () => { req.destroy(); reject(new Error('Request timed out')); });
    req.write(body);
    req.end();
  });
}

interface SearchChunk {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, string> | null;
}

async function searchAxon(apiBase: string, apiKey: string, query: string, topK: number, threshold?: number): Promise<SearchChunk[]> {
  const body: any = { query, top_k: topK };
  if (threshold != null) { body.threshold = threshold; }
  const result = await httpPost(`${apiBase}/search`, body, apiKey || undefined);
  if (result.status !== 200) {
    throw new Error(`Search returned HTTP ${result.status}: ${result.body}`);
  }
  const data = JSON.parse(result.body);
  // /search returns a plain array, not { results: [...] }
  return Array.isArray(data) ? data : (data.results ?? []);
}

async function startCopilotLlmWorker(apiBase: string, apiKey: string) {
  outputChannel.appendLine('Starting Copilot LLM Worker (Polling for tasks)...');
  while (true) {
    try {
      const result = await httpGet(`${apiBase}/llm/copilot/tasks`, apiKey);
      if (result.status === 200) {
        const data = JSON.parse(result.body);
        const tasks = data.tasks || [];

        if (tasks.length > 0) {
          outputChannel.appendLine(`Fulfilling ${tasks.length} Axon backend LLM tasks via Copilot...`);
          // Process in parallel
          await Promise.all(tasks.map(async (task: any) => {
            try {
              const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
              if (models.length === 0) throw new Error('Copilot LLM not available');

              const systemPrompt = task.system_prompt || 'You are a helpful assistant.';
              const messages = [
                vscode.LanguageModelChatMessage.User(systemPrompt + "\n\n" + task.prompt)
              ];

              // Use the first available model (usually GPT-4o or similar)
              const chatResponse = await models[0].sendRequest(messages, {});
              let fullText = '';
              for await (const chunk of chatResponse.text) {
                fullText += chunk;
              }

              await httpPost(`${apiBase}/llm/copilot/result/${task.id}`, { result: fullText }, apiKey);
            } catch (err) {
              outputChannel.appendLine(`Task ${task.id} failed: ${err}`);
              await httpPost(`${apiBase}/llm/copilot/result/${task.id}`, { error: String(err) }, apiKey);
            }
          }));
        }
      }
    } catch (err) {
      // Ignore connection errors during startup/shutdown
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// AxonStore share commands
// ---------------------------------------------------------------------------

async function initStore(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  let basePath = config.get<string>('storeBase', '');
  if (!basePath) {
    basePath = await vscode.window.showInputBox({
      prompt: 'Enter the base path for AxonStore (e.g. /data or ~/axon-data)',
      placeHolder: '/data',
    }) || '';
  }
  if (!basePath) { return; }
  try {
    const result = await httpPost(`${apiBase}/store/init`, { base_path: basePath }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Store init failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: AxonStore initialized at ${data.store_path} (user: ${data.username})`
    );
    outputChannel.appendLine(`AxonStore: ${data.store_path}, user dir: ${data.user_dir}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to initialize store. Is the server running?`);
  }
}

async function shareProject(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const project = await vscode.window.showInputBox({ prompt: 'Project name to share', placeHolder: 'research' });
  if (!project) { return; }
  const grantee = await vscode.window.showInputBox({ prompt: 'Grantee username (OS username on shared filesystem)', placeHolder: 'bob' });
  if (!grantee) { return; }
  const writeChoice = await vscode.window.showQuickPick(['Read-only', 'Read + Write'], { placeHolder: 'Access level' });
  const write_access = writeChoice === 'Read + Write';
  try {
    const result = await httpPost(`${apiBase}/share/generate`, { project, grantee, write_access }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Share generation failed — ${formatDetail(data, result.body)}`);
      return;
    }
    await vscode.env.clipboard.writeText(data.share_string);
    vscode.window.showInformationMessage(
      `Axon: Share key copied to clipboard (key: ${data.key_id}). Send the share string to ${grantee}.`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to generate share key.`);
  }
}

async function redeemShare(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const share_string = await vscode.window.showInputBox({
    prompt: 'Paste the share string you received from the project owner',
    placeHolder: 'base64 share string...',
  });
  if (!share_string) { return; }
  try {
    const result = await httpPost(`${apiBase}/share/redeem`, { share_string }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Redeem failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: Mounted "${data.owner}/${data.project}" as "${data.mount_name}" (${data.write_access ? 'read+write' : 'read-only'})`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to redeem share.`);
  }
}

async function revokeShare(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    const listResult = await httpGet(`${apiBase}/share/list`, apiKey);
    const data = JSON.parse(listResult.body);
    const active = (data.sharing || []).filter((s: any) => !s.revoked);
    if (active.length === 0) {
      vscode.window.showInformationMessage('Axon: No active shares to revoke.');
      return;
    }
    const items = active.map((s: any) => ({
      label: `${s.project} → ${s.grantee}`,
      description: `key: ${s.key_id} | ${s.write_access ? 'read+write' : 'read-only'}`,
      key_id: s.key_id,
    }));
    const picked = await vscode.window.showQuickPick(items, { placeHolder: 'Select share to revoke' });
    if (!picked) { return; }
    const revokeResult = await httpPost(`${apiBase}/share/revoke`, { key_id: (picked as any).key_id }, apiKey);
    const revokeData = JSON.parse(revokeResult.body);
    if (revokeResult.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Revoke failed — ${revokeData.detail || revokeResult.body}`);
      return;
    }
    vscode.window.showInformationMessage(`Axon: Revoked access for ${revokeData.grantee} to ${revokeData.project}.`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to revoke share.`);
  }
}

async function listShares(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    const result = await httpGet(`${apiBase}/share/list`, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: List shares failed.`);
      return;
    }
    const sharing = (data.sharing || []).map((s: any) =>
      `  • ${s.project} → ${s.grantee} [${s.write_access ? 'rw' : 'ro'}]${s.revoked ? ' (revoked)' : ''}`
    ).join('\n') || '  (none)';
    const shared = (data.shared || []).map((s: any) =>
      `  • ${s.owner}/${s.project} mounted as ${s.mount} [${s.write_access ? 'rw' : 'ro'}]`
    ).join('\n') || '  (none)';
    outputChannel.show();
    outputChannel.appendLine(`\n=== Axon Shares ===\nSharing with others:\n${sharing}\n\nShared with me:\n${shared}\n`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to list shares.`);
  }
}

async function refreshIngest(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    vscode.window.showInformationMessage('Axon: Checking for changed documents…');
    const result = await httpPost(`${apiBase}/ingest/refresh`, {}, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Refresh failed — ${formatDetail(data, result.body)}`);
      return;
    }
    const reingested = (data.reingested || []).length;
    const skipped = (data.skipped || []).length;
    const missing = (data.missing || []).length;
    const errors = (data.errors || []).length;
    outputChannel.show();
    outputChannel.appendLine(`\n=== Axon Refresh ===`);
    outputChannel.appendLine(`Re-ingested: ${reingested}  |  Unchanged: ${skipped}  |  Missing: ${missing}  |  Errors: ${errors}`);
    if (reingested > 0) {
      outputChannel.appendLine('Updated:');
      (data.reingested || []).forEach((s: string) => outputChannel.appendLine(`  ${s}`));
    }
    if (errors > 0) {
      outputChannel.appendLine('Errors:');
      (data.errors || []).forEach((e: any) => outputChannel.appendLine(`  ${e.source || e}: ${e.error || ''}`));
    }
    vscode.window.showInformationMessage(
      `Axon: Refresh complete — ${reingested} updated, ${skipped} unchanged${missing ? `, ${missing} missing` : ''}`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Refresh failed. Is the server running?`);
  }
}

async function listStaleDocs(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const daysInput = await vscode.window.showInputBox({
    prompt: 'Show documents not refreshed in N days',
    value: '7',
    placeHolder: '7',
  });
  if (daysInput === undefined) { return; }
  const days = parseInt(daysInput, 10) || 7;
  try {
    const result = await httpGet(`${apiBase}/collection/stale?days=${days}`, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Stale list failed — ${formatDetail(data, result.body)}`);
      return;
    }
    const stale: any[] = data.stale_docs || [];
    outputChannel.show();
    outputChannel.appendLine(`\n=== Axon Stale Docs (>${days} days) ===`);
    if (stale.length === 0) {
      outputChannel.appendLine('All documents are fresh.');
      vscode.window.showInformationMessage(`Axon: All documents are fresh (threshold: ${days} days).`);
    } else {
      stale.sort((a, b) => b.age_days - a.age_days);
      stale.forEach(s => outputChannel.appendLine(`  ${String(s.age_days).padStart(5)}d  [${s.project}]  ${s.doc_id}`));
      outputChannel.appendLine(`Total: ${stale.length}`);
      vscode.window.showInformationMessage(`Axon: ${stale.length} stale document(s) found. Check Axon output panel.`);
    }
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to list stale documents.`);
  }
}

async function clearKnowledgeBase(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const confirmed = await vscode.window.showWarningMessage(
    'Axon: Clear the entire knowledge base for the current project? This cannot be undone.',
    { modal: true },
    'Clear Knowledge Base',
  );
  if (confirmed !== 'Clear Knowledge Base') { return; }
  try {
    const result = await httpPost(`${apiBase}/clear`, {}, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Clear failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage('Axon: Knowledge base cleared for current project.');
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to clear knowledge base.`);
  }
}

async function showGraphStatus(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    const result = await httpGet(`${apiBase}/graph/status`, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Graph status failed — ${formatDetail(data, result.body)}`);
      return;
    }
    const inProgress = data.community_build_in_progress;
    const count = data.community_summary_count;
    outputChannel.show();
    outputChannel.appendLine(`\n=== Axon GraphRAG Status ===`);
    outputChannel.appendLine(`Community summaries: ${count}`);
    outputChannel.appendLine(`Build in progress:   ${inProgress ? 'yes' : 'no'}`);
    vscode.window.showInformationMessage(
      `Axon GraphRAG: ${count} community summaries${inProgress ? ' (build in progress)' : ''}`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to get graph status.`);
  }
}

// ---------------------------------------------------------------------------
// AxonStore LM Tools
// ---------------------------------------------------------------------------

class AxonListSharesTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpGet(`${apiBase}/share/list`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Share list error: ${formatDetail(data, result.body)}`)]);
      }
      const text = JSON.stringify(data, null, 2);
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(text)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error listing shares: ${err}`)]);
    }
  }
}

// ---------------------------------------------------------------------------
// AxonStore initialisation tool
// ---------------------------------------------------------------------------

class AxonInitStoreTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: `Initialising AxonStore at "${options.input.base_path}"...` };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { base_path } = options.input;
    try {
      const result = await httpPost(`${apiBase}/store/init`, { base_path }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`AxonStore init error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`AxonStore initialised at ${data.store_path} (user: ${data.username}). Share tools are now available.`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error initialising AxonStore: ${err}`)]);
    }
  }
}

// ---------------------------------------------------------------------------
// Image ingestion via Copilot multimodal vision
// ---------------------------------------------------------------------------

/** Map file extension → MIME type for image ingest. */
function getImageMimeType(filePath: string): string {
  const ext = filePath.split('.').pop()?.toLowerCase() ?? '';
  const mimeMap: Record<string, string> = {
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    bmp: 'image/bmp',
    tif: 'image/tiff',
    tiff: 'image/tiff',
    webp: 'image/webp',
  };
  return mimeMap[ext] ?? 'image/png';
}

const SUPPORTED_IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp']);

class AxonIngestImageTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return {
      invocationMessage: `Describing image "${options.input.imagePath}" with Copilot and ingesting into Axon...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const { imagePath, project } = options.input as { imagePath: string; project?: string };
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');

    // Validate extension
    const ext = imagePath.split('.').pop()?.toLowerCase() ?? '';
    if (!SUPPORTED_IMAGE_EXTENSIONS.has(ext)) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(
          `Unsupported image format ".${ext}". Supported: PNG, JPG, JPEG, BMP, TIF, TIFF, WEBP.`
        )
      ]);
    }

    try {
      // Read the image file as Uint8Array (LanguageModelDataPart requires Uint8Array, not Buffer)
      const fs = await import('fs');
      const rawBuffer = fs.readFileSync(imagePath);
      const imageBuffer = new Uint8Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);
      const mimeType = getImageMimeType(imagePath);

      // Select a Copilot model with vision capability
      const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
      if (models.length === 0) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart('No Copilot language model available. Ensure GitHub Copilot is installed and signed in.')
        ]);
      }

      // Prefer a model that supports image input; fall back to first available
      const model = models.find((m: any) => m.capabilities?.supportsImageToText) ?? models[0];

      // Build the multimodal prompt
      const prompt = [
        vscode.LanguageModelChatMessage.User([
          new (vscode as any).LanguageModelDataPart(imageBuffer, mimeType),
          new vscode.LanguageModelTextPart(
            'Describe this image in detail for a searchable knowledge base. ' +
            'Include all visible text, diagram structure, key concepts, labels, ' +
            'relationships, and any quantitative data shown. Be thorough and precise.'
          )
        ])
      ];

      // Stream the description
      const response = await model.sendRequest(prompt, {}, token);
      let description = '';
      for await (const chunk of response.text) {
        description += chunk;
      }

      if (!description.trim()) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart('Copilot returned an empty description for the image.')
        ]);
      }

      // Ingest the description as text
      const body: Record<string, any> = {
        text: description,
        metadata: { type: 'image', original_path: imagePath, ingested_via: 'copilot_vision' }
      };
      if (project) {
        body['project'] = project;
      }
      const result = await httpPost(`${apiBase}/add_text`, body, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(`Axon Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)
        ]);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(
          `Image ingested successfully. doc_id: ${data.doc_id}, status: ${data.status}. ` +
          `Description (${description.length} chars) stored in Axon.`
        )
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Error during image ingest: ${err}`)
      ]);
    }
  }
}

// ---------------------------------------------------------------------------
// Graph panel LM Tool
// ---------------------------------------------------------------------------

class AxonShowGraphTool implements vscode.LanguageModelTool<any> {
  constructor(private readonly context: vscode.ExtensionContext) {}

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: `Opening Axon graph for: "${options.input.query}"…` };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const { query } = options.input;
    try {
      const status = await showGraphForQuery(this.context, query);
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Graph panel status: ${status}`)
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Error opening graph panel: ${err}`)
      ]);
    }
  }
}
