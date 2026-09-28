/**
 * Graph LM tools and VS Code command implementations.
 */

import * as vscode from 'vscode';

import { state, resolveApiBase } from '../shared';

import { httpGet, httpPost, formatDetail, parseJsonSafe, apiConnectionError } from '../client/http';

import { showGraphForQuery } from '../graph/panel';

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
        new (vscode as any).LanguageModelTextPart(`Error opening graph panel: ${err instanceof Error ? err.message : String(err)}`)
      ]);
    }
  }

}

export class AxonGraphRetrieveTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    const q = options.input && options.input.query ? String(options.input.query) : '';
    return { invocationMessage: `Running graph backend retrieve for: "${q}"…` };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = resolveApiBase();
    const apiKey = config.get<string>('apiKey', '');
    const body: any = { query: options.input?.query ?? '' };
    if (typeof options.input?.top_k === 'number') body.top_k = options.input.top_k;
    if (typeof options.input?.point_in_time === 'string') body.point_in_time = options.input.point_in_time;
    if (options.input?.federation_weights && typeof options.input.federation_weights === 'object') {
      body.federation_weights = options.input.federation_weights;
    }
    // Assertion, not a switch: 409 if Axon is serving a different project.
    if (typeof options.input?.project === 'string' && options.input.project) { body.project = options.input.project; }
    try {
      const result = await httpPost(`${apiBase}/graph/retrieve`, body, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Graph retrieve error: ${formatDetail(data, result.body)}`)]);
      }
      const ctxs = Array.isArray(data.contexts) ? data.contexts : [];
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Backend '${data.backend ?? 'unknown'}' returned ${ctxs.length} context(s).\n${JSON.stringify(data, null, 2)}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }

}

/**
 * update_fact — assert or correct one graph fact (mirrors the MCP tool and
 * POST /graph/facts). Only the fields the model supplied are sent: the REST
 * body is extra=forbid, and omitted `replace` means "backend default".
 */
export class AxonUpdateFactTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    const { subject, relation, object } = options.input ?? {};
    return { invocationMessage: `Recording fact: ${subject} ${relation} ${object}…` };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = resolveApiBase();
    const apiKey = config.get<string>('apiKey', '');
    const input = options.input ?? {};
    const body: any = { subject: input.subject, relation: input.relation, object: input.object };
    if (typeof input.description === 'string' && input.description) { body.description = input.description; }
    if (typeof input.confidence === 'number') { body.confidence = input.confidence; }
    if (typeof input.replace === 'boolean') { body.replace = input.replace; }
    if (typeof input.project === 'string' && input.project) { body.project = input.project; }
    try {
      const result = await httpPost(`${apiBase}/graph/facts`, body, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Update fact error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      let msg = `Fact ${data.status ?? 'updated'}`;
      if (data.fact_id) { msg += ` (fact_id ${data.fact_id})`; }
      if (Array.isArray(data.superseded_ids) && data.superseded_ids.length) { msg += `; superseded ${data.superseded_ids.length}`; }
      if (Array.isArray(data.conflicted_ids) && data.conflicted_ids.length) { msg += `; conflicts with ${data.conflicted_ids.length}`; }
      if (data.detail) { msg += ` — ${data.detail}`; }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(msg)]);
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

