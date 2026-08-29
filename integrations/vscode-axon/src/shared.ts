/**
 * Shared mutable extension state.
 *
 * All modules that need access to serverProcess, externalServerPid, or
 * outputChannel import this object.  Mutating properties on `state` is visible
 * to every other module because CommonJS caches the module instance.
 */
import * as vscode from 'vscode';
import { ChildProcess } from 'child_process';

export const state = {
  serverProcess: undefined as ChildProcess | undefined,
  externalServerPid: undefined as number | undefined,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  outputChannel: undefined as unknown as vscode.OutputChannel,
  // The actual resolved server address — set once by ensureServerRunning()
  // (explicit user override, an adopted still-alive server, or a freshly
  // chosen free port) and read by every command/tool from then on via
  // resolveApiBase(), instead of each independently re-reading the static
  // axon.apiBase setting (which no longer reflects the real port once it's
  // chosen dynamically).
  apiBase: '' as string,
};

export const SERVER_START_TIMEOUT_MS = 120_000; // >= 1 min; startup may include model warmup
export const GRAPH_ANSWER_TIMEOUT_MS = 60_000;  // query synthesis can be slower than regular API calls

/**
 * The effective Axon API base URL: state.apiBase once ensureServerRunning()
 * has resolved it, otherwise a defensive fallback to the raw axon.apiBase
 * setting (empty unless the user explicitly configured one) or the classic
 * default — covers the edge case where a command runs before the server
 * lifecycle has resolved anything (e.g. axon.autoStart is off and the user
 * never ran "Axon: Start Server").
 */
export function resolveApiBase(): string {
  if (state.apiBase) {
    return state.apiBase;
  }
  const configured = vscode.workspace.getConfiguration('axon').get<string>('apiBase', '');
  return configured || 'http://127.0.0.1:8420';
}
