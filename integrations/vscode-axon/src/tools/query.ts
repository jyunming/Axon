/**
 * Query and search LM tools + chat participant handler.
 */
import * as vscode from 'vscode';

import { state } from '../shared';
import { httpPost, searchAxon, formatDetail } from '../client/http';

export class AxonSearchTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return {
      invocationMessage: `Searching Axon knowledge base for: "${options.input.query}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { query, topK = 5, threshold, filters, project } = options.input;

    try {
      let chunks = await searchAxon(apiBase, apiKey, query, topK, threshold, filters, project);
      // When a threshold filtered out all results, fall back to top-N without the threshold
      // so the caller gets candidates rather than an empty response.
      if (chunks.length === 0 && threshold != null) {
        const fallback = await searchAxon(apiBase, apiKey, query, Math.min(topK, 3), undefined, filters, project);
        if (fallback.length > 0) {
          const scores = fallback.map(c => {
            const s = (c as any).score;
            return `score ${typeof s === 'number' && Number.isFinite(s) ? s.toFixed(3) : '?'}`;
          }).join(', ');
          const note = `*No results met the threshold (${threshold}). Showing top candidates (${scores}):*\n\n`;
          const content = note + fallback.map(c => `[ID: ${c.id}] Source: ${c.metadata?.source}\n${c.text}`).join('\n\n---\n\n');
          return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(content)]);
        }
      }
      const content = chunks.map(c => `[ID: ${c.id}] Source: ${c.metadata?.source}\n${c.text}`).join('\n\n---\n\n');
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(content || 'No results found.')]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during Axon search: ${err}`)]);
    }
  }
}

export class AxonQueryTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return {
      invocationMessage: `Querying Axon knowledge base: "${options.input.query}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { query, top_k, filters, project } = options.input;

    try {
      // discuss: false disables the general-knowledge fallback so the tool
      // returns a definitive "no data" message instead of a hallucinated answer
      // when the knowledge base has no matching content.
      const body: any = { query, discuss: false };
      if (top_k != null) { body.top_k = top_k; }
      if (filters != null) { body.filters = filters; }
      if (project != null) { body.project = project; }
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

export async function chatHandler(
  request: vscode.ChatRequest,
  _context: vscode.ChatContext,
  response: vscode.ChatResponseStream,
  _token: vscode.CancellationToken,
): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const apiKey = config.get<string>('apiKey', '');

  // Route through the same /query backend as axon_queryKnowledge for consistent behavior.
  // discuss: false prevents the general-knowledge fallback from producing answers
  // when the knowledge base contains no matching content.
  try {
    const result = await httpPost(`${apiBase}/query`, { query: request.prompt, discuss: false }, apiKey, 20_000, 'vscode_chat_participant');
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      response.markdown(`> **Axon error** (${result.status}): ${formatDetail(data, result.body)}`);
      return;
    }
    const answer = data.response || '*No answer generated — the knowledge base may not contain relevant content.*';
    state.outputChannel.appendLine(`@axon query: "${request.prompt}"`);
    response.markdown(answer);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    response.markdown(`> **Axon error**: Could not reach the API at \`${apiBase}\`. ${msg}\n\nMake sure the Axon server is running (or enable \`axon.autoStart\`).`);
  }
}
