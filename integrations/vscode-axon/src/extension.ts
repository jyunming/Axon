import * as vscode from 'vscode';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';

let serverProcess: ChildProcess | undefined;
let externalServerPid: number | undefined; // PID of a server we didn't spawn but own on deactivate
let outputChannel: vscode.OutputChannel;

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
    waitForHealth(apiBase, 15000).then((running) => {
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
        (vscode as any).lm.registerTool('axon_listProjects', new AxonListProjectsTool()),
        (vscode as any).lm.registerTool('axon_switchProject', new AxonSwitchProjectTool()),
        (vscode as any).lm.registerTool('axon_createProject', new AxonCreateProjectTool()),
        (vscode as any).lm.registerTool('axon_deleteProject', new AxonDeleteProjectTool()),
        (vscode as any).lm.registerTool('axon_deleteDocuments', new AxonDeleteDocumentsTool()),
        (vscode as any).lm.registerTool('axon_getCollection', new AxonGetCollectionTool()),
        (vscode as any).lm.registerTool('axon_clearCollection', new AxonClearCollectionTool()),
        (vscode as any).lm.registerTool('axon_updateSettings', new AxonUpdateSettingsTool()),
        (vscode as any).lm.registerTool('axon_listShares', new AxonListSharesTool()),
        (vscode as any).lm.registerTool('axon_ingestImage', new AxonIngestImageTool())
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
      const body: any = { query };
      if (top_k != null) { body.top_k = top_k; }
      const result = await httpPost(`${apiBase}/query`, body, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Ingest Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon URL Ingest Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Path Ingest Error (${result.status}): ${data.detail || result.body}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Message: ${data.message}, JobID: ${data.job_id}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during path ingest: ${err}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Project Creation Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Project Deletion Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Document Deletion Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Clear Error (${result.status}): ${data.detail || result.body}`)]);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Configuration Error (${result.status}): ${data.detail || result.body}`)]);
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
  const pythonPath = config.get<string>('pythonPath', 'python');

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

  const started = await waitForHealth(apiBase, 30_000); // Increased to 30s

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
      vscode.window.showErrorMessage(`Axon: Ingest failed — ${data.detail || result.body}`);
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
      vscode.window.showErrorMessage(`Axon: Ingest failed — ${data.detail || result.body}`);
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
      vscode.window.showErrorMessage(`Axon: Store init failed — ${data.detail || result.body}`);
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
      vscode.window.showErrorMessage(`Axon: Share generation failed — ${data.detail || result.body}`);
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
      vscode.window.showErrorMessage(`Axon: Redeem failed — ${data.detail || result.body}`);
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
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Share list error: ${data.detail || result.body}`)]);
      }
      const text = JSON.stringify(data, null, 2);
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(text)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error listing shares: ${err}`)]);
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
      // Read the image file
      const fs = await import('fs');
      const imageBuffer = fs.readFileSync(imagePath);
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
          new (vscode as any).LanguageModelDataPart(mimeType, imageBuffer),
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
          new (vscode as any).LanguageModelTextPart(`Axon Ingest Error (${result.status}): ${data.detail || result.body}`)
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
