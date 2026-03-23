/**
 * Graph LM tools and VS Code command implementations.
 */
import * as vscode from 'vscode';

import { state } from '../shared';
import { httpGet, httpPost, formatDetail, apiConnectionError } from '../client/http';
import { showGraphForQuery } from '../graph/panel';

export class AxonShowGraphStatusTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Fetching GraphRAG status…' };
  }

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpGet(`${apiBase}/graph/status`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Graph status error: ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(JSON.stringify(data, null, 2))]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}

export class AxonShowGraphTool implements vscode.LanguageModelTool<any> {
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

export class AxonFinalizeGraphTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Finalizing Axon knowledge graph (community rebuild)…' };
  }

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpPost(`${apiBase}/graph/finalize`, {}, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Graph finalize error: ${formatDetail(data, result.body)}`)]);
      }
      const summaries = data.community_summary_count ?? data.summaries ?? '?';
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Graph finalized. Community summaries: ${summaries}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}

export async function showGraphStatus(apiBase: string): Promise<void> {
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
    state.outputChannel.show();
    state.outputChannel.appendLine(`\n=== Axon GraphRAG Status ===`);
    state.outputChannel.appendLine(`Community summaries: ${count}`);
    state.outputChannel.appendLine(`Build in progress:   ${inProgress ? 'yes' : 'no'}`);
    vscode.window.showInformationMessage(
      `Axon GraphRAG: ${count} community summaries${inProgress ? ' (build in progress)' : ''}`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to get graph status. ${apiConnectionError(err)}`);
  }
}
