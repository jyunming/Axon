/**
 * Server lifecycle — spawn, stop, health-check, Copilot LLM worker.
 */
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { spawn } from 'child_process';

import { state, SERVER_START_TIMEOUT_MS } from '../shared';
import { httpGet, httpPost, sleep } from './http';

export async function getPortPid(port: number): Promise<number | undefined> {
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
export async function discoverPythonPath(): Promise<string> {
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
      state.outputChannel.appendLine(`Python auto-detected via ~/.axon/.python_path: ${discovered}`);
      return discovered;
    }
  }
  // 3. pipx isolated venv (fixed path for `pipx install axon`)
  const pipxPython = isWin
    ? path.join(os.homedir(), '.local', 'pipx', 'venvs', 'axon', 'Scripts', 'python.exe')
    : path.join(os.homedir(), '.local', 'pipx', 'venvs', 'axon', 'bin', 'python');
  if (fs.existsSync(pipxPython)) {
    state.outputChannel.appendLine(`Python auto-detected via pipx venv: ${pipxPython}`);
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
          state.outputChannel.appendLine(`Python auto-detected via workspace venv (${venvDir}): ${candidate}`);
          return candidate;
        }
      }
    }
  }
  // 5. System Python fallback
  const systemPython = isWin ? 'python' : 'python3';
  state.outputChannel.appendLine(
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

export async function ensureServerRunning(apiBase: string, context: vscode.ExtensionContext): Promise<void> {
  const portMatch = apiBase.match(/:(\d+)/);
  const port = portMatch ? parseInt(portMatch[1], 10) : 8000;
  if (await isAxonRunning(apiBase)) {
    state.outputChannel.appendLine('Axon API already running.');
    // Capture PID so we can stop it on deactivate even if we didn't spawn it
    state.externalServerPid = await getPortPid(port);
    if (state.externalServerPid) {
      state.outputChannel.appendLine(`Tracking external server PID: ${state.externalServerPid}`);
    }
    return;
  }
  // Detect stale listener: something is bound to the port but not answering /health.
  // Only auto-kill if the process can be positively identified as an Axon/uvicorn process
  // to avoid terminating unrelated user services bound to the same port.
  const stalePid = await getPortPid(port);
  if (stalePid) {
    let isAxonProcess = false;
    try {
      const { execSync } = require('child_process');
      let cmdLine = '';
      if (process.platform === 'win32') {
        // Get the full command line (not just exe path) so we can match axon.api signals
        cmdLine = execSync(
          `powershell -Command "(Get-CimInstance Win32_Process -Filter 'ProcessId=${stalePid}').CommandLine"`,
          { encoding: 'utf8' }
        ).trim().toLowerCase();
      } else {
        cmdLine = execSync(`ps -p ${stalePid} -o command=`, { encoding: 'utf8' }).trim().toLowerCase();
      }
      // Only match Axon/uvicorn-specific signals — avoid the generic "python" substring
      const axonSignals = ['axon.api', 'uvicorn axon.api:app', 'python -m axon.api', 'axon-api'];
      isAxonProcess = axonSignals.some(signal => cmdLine.includes(signal));
    } catch {
      // If we can't inspect the process, don't kill it
    }
    if (isAxonProcess) {
      state.outputChannel.appendLine(
        `Axon: stale process (PID ${stalePid}) found on port ${port} — terminating and restarting.`
      );
      try {
        // Attempt graceful shutdown first; escalate to force-kill only if needed
        if (process.platform === 'win32') {
          require('child_process').execSync('taskkill /PID ' + stalePid);
        } else {
          process.kill(stalePid, 'SIGTERM');
        }
        await sleep(1500);
        // Verify it's gone; force-kill if still running
        const stillRunning = await getPortPid(port);
        if (stillRunning === stalePid) {
          if (process.platform === 'win32') {
            require('child_process').execSync('taskkill /F /PID ' + stalePid);
          } else {
            process.kill(stalePid, 'SIGKILL');
          }
          await sleep(500);
        }
      } catch {
        state.outputChannel.appendLine(
          `Could not terminate stale process ${stalePid} — it may have already exited or access was denied.`
        );
      }
    } else {
      state.outputChannel.appendLine(
        `Axon: port ${port} is in use by a non-Axon process (PID ${stalePid}). ` +
        `Cannot auto-start — free the port or change axon.apiBase to a different port.`
      );
      vscode.window.showWarningMessage(
        `Axon: port ${port} is already in use by another process (PID ${stalePid}). ` +
        `Free the port or update the axon.apiBase setting.`
      );
      return;
    }
  }
  const config = vscode.workspace.getConfiguration('axon');
  const pythonPath = await discoverPythonPath();
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    state.outputChannel.appendLine('No workspace folder open. Cannot auto-start Axon server.');
    return;
  }
  const workspaceRoot = workspaceFolders[0].uri.fsPath;
  // Port as string for uvicorn spawn argument
  let portStr = String(port);
  // Default ingest base to workspaceRoot for safety (prevents ingesting files outside the project).
  // Users can broaden this to the filesystem root via axon.ingestBase in settings.
  const configuredBase = config.get<string>('ingestBase', '');
  const fsRoot = configuredBase || workspaceRoot;
  const storeBase = config.get<string>('storeBase', '');
  state.outputChannel.appendLine(`Starting Axon API server with: ${pythonPath} -m uvicorn axon.api:app --host 127.0.0.1 --port ${portStr}`);
  state.outputChannel.appendLine(`RAG_INGEST_BASE=${fsRoot} (any path under this root is ingestable)`);
  if (storeBase) {
    state.outputChannel.appendLine(`AXON_STORE_BASE=${storeBase}`);
  }
  state.serverProcess = spawn(pythonPath, ['-m', 'uvicorn', 'axon.api:app', '--host', '127.0.0.1', '--port', portStr], {
    cwd: workspaceRoot,
    shell: process.platform === 'win32', // Required on Windows for Python path resolution
    env: {
      ...process.env,
      PYTHONPATH: path.join(workspaceRoot, 'src'),
      RAG_INGEST_BASE: fsRoot,
      ...(storeBase ? { AXON_STORE_BASE: storeBase } : {}),
    },
  });
  state.serverProcess.on('error', (err) => {
    state.outputChannel.appendLine(`Axon server spawn error: ${err.message}`);
    state.serverProcess = undefined;
  });
  state.serverProcess.stdout?.on('data', (data: Buffer) => {
    state.outputChannel.append(`[server] ${data.toString()}`);
  });
  state.serverProcess.stderr?.on('data', (data: Buffer) => {
    state.outputChannel.append(`[server] ${data.toString()}`);
  });
  state.serverProcess.on('exit', (code) => {
    state.outputChannel.appendLine(`Axon server exited with code ${code}`);
    state.serverProcess = undefined;
  });
  const started = await waitForHealth(apiBase, SERVER_START_TIMEOUT_MS);
  if (started) {
    state.outputChannel.appendLine('Axon API server is ready.');
    vscode.window.showInformationMessage('Axon API server started successfully.');
  } else {
    state.outputChannel.appendLine(`Axon API server did not become ready within ${Math.round(SERVER_START_TIMEOUT_MS / 1000)} seconds.`);
    vscode.window.showWarningMessage('Axon API server failed to start. Check the Axon output panel.');
  }
}

export function stopServer(): void {
  if (state.serverProcess) {
    state.outputChannel.appendLine('Stopping Axon API server...');
    state.serverProcess.kill();
    state.serverProcess = undefined;
  } else if (state.externalServerPid) {
    state.outputChannel.appendLine(`Stopping external Axon API server (PID ${state.externalServerPid})...`);
    try {
      process.kill(state.externalServerPid);
    } catch {
      // Process may have already exited
    }
    state.externalServerPid = undefined;
  }
}

export async function isAxonRunning(apiBase: string): Promise<boolean> {
  try {
    const result = await httpGet(`${apiBase}/health`);
    return result.status === 200;
  } catch {
    return false;
  }
}

export async function waitForHealth(apiBase: string, timeoutMs: number): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await isAxonRunning(apiBase)) {
      return true;
    }
    // Fail fast if the process we spawned has crashed
    if (state.serverProcess === undefined) {
      break;
    }
    await sleep(500);
  }
  return false;
}

export async function startCopilotLlmWorker(apiBase: string, apiKey: string): Promise<void> {
  state.outputChannel.appendLine('Starting Copilot LLM Worker (Polling for tasks)...');
  while (true) {
    try {
      const result = await httpGet(`${apiBase}/llm/copilot/tasks`, apiKey);
      if (result.status === 200) {
        const data = JSON.parse(result.body);
        const tasks = data.tasks || [];
        if (tasks.length > 0) {
          state.outputChannel.appendLine(`Fulfilling ${tasks.length} Axon backend LLM tasks via Copilot...`);
          // Process in parallel
          await Promise.all(tasks.map(async (task: any) => {
            try {
              const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
              if (models.length === 0) { throw new Error('Copilot LLM not available'); }
              const systemPrompt = task.system_prompt || 'You are a helpful assistant.';
              const messages = [
                vscode.LanguageModelChatMessage.User(systemPrompt + '\n\n' + task.prompt)
              ];
              // Use the first available model (usually GPT-4o or similar)
              const chatResponse = await models[0].sendRequest(messages, {});
              let fullText = '';
              for await (const chunk of chatResponse.text) {
                fullText += chunk;
              }
              await httpPost(`${apiBase}/llm/copilot/result/${task.id}`, { result: fullText }, apiKey);
            } catch (err) {
              state.outputChannel.appendLine(`Task ${task.id} failed: ${err}`);
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
