/**
 * Graph LM tools and VS Code command implementations.
 */

import * as vscode from 'vscode';

import { state } from '../shared';

import { httpGet, httpPost, formatDetail, apiConnectionError } from '../client/http';

import { showGraphForQuery } from '../graph/panel';

export class AxonGraphStatusTool implements vscode.LanguageModelTool<any> {
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
        new (vscode as any).LanguageModelTextPart(`Error opening graph panel: ${err instanceof Error ? err.message : String(err)}`)
      ]);
    }
  }

}

export class AxonGraphFinalizeTool implements vscode.LanguageModelTool<any> {
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
      const status = data.status ?? 'ok';
      const summaries = data.community_summary_count ?? data.summaries ?? 0;
      const detail = data.detail ?? '';
      const backendId = data.backend_id ?? '';
      let msg: string;
      if (status === 'not_applicable') {
        msg = `Graph finalize not applicable on backend '${backendId || 'unknown'}'${detail ? ` — ${detail}` : ''}.`;
      } else {
        msg = `Graph finalized. Community summaries: ${summaries}.`;
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(msg)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }

}

export class AxonGraphConflictsTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Listing conflicted graph facts…' };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const limit = (options.input && typeof options.input.limit === 'number') ? options.input.limit : 100;
    try {
      const result = await httpGet(`${apiBase}/graph/conflicts?limit=${encodeURIComponent(String(limit))}`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Graph conflicts error: ${formatDetail(data, result.body)}`)]);
      }
      if (data.supported === false) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Backend '${data.backend ?? 'unknown'}' does not track conflicted facts.`)]);
      }
      const conflicts = Array.isArray(data.conflicts) ? data.conflicts : [];
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Backend '${data.backend ?? 'unknown'}' — ${conflicts.length} conflict(s):\n${JSON.stringify(conflicts, null, 2)}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
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
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const body: any = { query: options.input?.query ?? '' };
    if (typeof options.input?.top_k === 'number') body.top_k = options.input.top_k;
    if (typeof options.input?.point_in_time === 'string') body.point_in_time = options.input.point_in_time;
    if (options.input?.federation_weights && typeof options.input.federation_weights === 'object') {
      body.federation_weights = options.input.federation_weights;
    }
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

export class AxonGraphDataTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Fetching Axon knowledge graph data…' };
  }
  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpGet(`${apiBase}/graph/data`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Graph data error: ${formatDetail(data, result.body)}`)]);
      }
      const MAX_NODES = 500;
      const MAX_LINKS = 1000;
      const originalNodes = Array.isArray(data.nodes) ? data.nodes : [];
      const originalLinks = Array.isArray(data.links) ? data.links : [];
      const nodeCount = originalNodes.length;
      const linkCount = originalLinks.length;
      const truncatedNodes = originalNodes.slice(0, MAX_NODES);
      const truncatedLinks = originalLinks.slice(0, MAX_LINKS);
      const truncated =
        truncatedNodes.length < originalNodes.length ||
        truncatedLinks.length < originalLinks.length;
      const responsePayload = { ...data, nodes: truncatedNodes, links: truncatedLinks, truncated };
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Graph: ${nodeCount} nodes, ${linkCount} edges.\n${JSON.stringify(responsePayload, null, 2)}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }

}

export class AxonGetActiveLeasesTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Fetching active write leases…' };
  }
  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpGet(`${apiBase}/registry/leases`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Leases error: ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(JSON.stringify(data, null, 2))]);
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

