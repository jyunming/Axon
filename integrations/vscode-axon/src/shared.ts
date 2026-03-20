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
};

export const SERVER_START_TIMEOUT_MS = 120_000; // >= 1 min; startup may include model warmup
export const GRAPH_ANSWER_TIMEOUT_MS = 60_000;  // query synthesis can be slower than regular API calls
