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

export async function chatHandler(
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
      state.outputChannel.appendLine(`Retrieved ${chunks.length} chunks for query: "${request.prompt}"`);
    } else {
      state.outputChannel.appendLine('No chunks found for query.');
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
