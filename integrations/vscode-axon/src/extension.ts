import * as vscode from 'vscode';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';

let serverProcess: ChildProcess | undefined;
let outputChannel: vscode.OutputChannel;

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  outputChannel = vscode.window.createOutputChannel('Axon');
  context.subscriptions.push(outputChannel);

  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://localhost:8000');
  const autoStart = config.get<boolean>('autoStart', true);

  outputChannel.appendLine(`Axon extension activating. API base: ${apiBase}`);

  if (autoStart) {
    await ensureServerRunning(apiBase, context);
  }

  // Register the @axon chat participant
  const participant = vscode.chat.createChatParticipant('axon.chat', chatHandler);
  participant.iconPath = new vscode.ThemeIcon('database');
  context.subscriptions.push(participant);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('axon.switchProject', () => switchProject(apiBase)),
    vscode.commands.registerCommand('axon.ingestFile', () => ingestCurrentFile(apiBase)),
    vscode.commands.registerCommand('axon.startServer', () => ensureServerRunning(apiBase, context)),
    vscode.commands.registerCommand('axon.stopServer', () => stopServer()),
  );

  outputChannel.appendLine('Axon extension ready.');
}

export function deactivate(): void {
  stopServer();
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

async function ensureServerRunning(apiBase: string, context: vscode.ExtensionContext): Promise<void> {
  if (await isAxonRunning(apiBase)) {
    outputChannel.appendLine('Axon API already running.');
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

  outputChannel.appendLine(`Starting Axon API server with: ${pythonPath} -m uvicorn axon.api:app --host 127.0.0.1 --port ${port}`);

  serverProcess = spawn(pythonPath, ['-m', 'uvicorn', 'axon.api:app', '--host', '127.0.0.1', '--port', port], {
    cwd: workspaceRoot,
    env: {
      ...process.env,
      PYTHONPATH: path.join(workspaceRoot, 'src'),
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

  const started = await waitForHealth(apiBase, 15_000);
  if (started) {
    outputChannel.appendLine('Axon API server is ready.');
    vscode.window.showInformationMessage('Axon API server started successfully.');
  } else {
    outputChannel.appendLine('Axon API server did not become ready within 15 seconds.');
    vscode.window.showWarningMessage('Axon API server failed to start. Check the Axon output panel.');
  }
}

function stopServer(): void {
  if (serverProcess) {
    outputChannel.appendLine('Stopping Axon API server...');
    serverProcess.kill();
    serverProcess = undefined;
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
  const apiBase = config.get<string>('apiBase', 'http://localhost:8000');
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

  let projects: string[] = [];
  try {
    const result = await httpGet(`${apiBase}/projects`, apiKey);
    const data = JSON.parse(result.body) as { projects?: string[] };
    projects = data.projects ?? [];
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to list projects. Is the server running?`);
    return;
  }

  if (projects.length === 0) {
    vscode.window.showInformationMessage('Axon: No projects found.');
    return;
  }

  const selected = await vscode.window.showQuickPick(projects, {
    placeHolder: 'Select an Axon project',
  });
  if (!selected) {
    return;
  }

  try {
    await httpPost(`${apiBase}/project/switch`, { project: selected }, apiKey);
    vscode.window.showInformationMessage(`Axon: Switched to project "${selected}".`);
    outputChannel.appendLine(`Switched to project: ${selected}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to switch project.`);
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
    req.setTimeout(10_000, () => { req.destroy(); reject(new Error('Request timed out')); });
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

async function searchAxon(apiBase: string, apiKey: string, query: string, topK: number): Promise<SearchChunk[]> {
  const result = await httpPost(`${apiBase}/search`, { query, top_k: topK }, apiKey || undefined);
  if (result.status !== 200) {
    throw new Error(`Search returned HTTP ${result.status}: ${result.body}`);
  }
  const data = JSON.parse(result.body) as { results?: SearchChunk[] };
  return data.results ?? [];
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
