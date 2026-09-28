/**
 * Config LM tools (get_config / set_config) + the axon.configSetup wizard command.
 */

import * as vscode from 'vscode';

import { state, resolveApiBase } from '../shared';

import { httpGet, httpPost, formatDetail, parseJsonSafe, apiConnectionError } from '../client/http';

// ---------------------------------------------------------------------------

// LM Tools

// ---------------------------------------------------------------------------

function formatValidation(data: any): string {
  const issues: any[] = data?.issues || [];
  const valid: boolean = data?.valid ?? true;
  if (issues.length === 0) {
    return 'Config validation passed. No issues found.';
  }
  const lines = issues.map((issue: any) => {
    const suggestion = issue.suggestion ? ` Suggestion: ${issue.suggestion}` : '';
    return `[${String(issue.level).toUpperCase()}] ${issue.section}.${issue.field}: ${issue.message}${suggestion}`;
  });
  const summary = valid
    ? `Config has ${issues.length} notice(s) (no errors):`
    : `Config has errors (${issues.filter((i: any) => i.level === 'error').length} error(s)):`;
  return `${summary}\n${lines.join('\n')}`;
}

/** get_config — GET /config (secrets masked); validate=true adds GET /config/validate. */
export class AxonConfigGetTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return {
      invocationMessage: options.input?.validate ? 'Reading and validating Axon config…' : 'Reading Axon config…',
    };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = resolveApiBase();
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpGet(`${apiBase}/config`, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`),
        ]);
      }
      let text = `Current Axon config:\n${JSON.stringify(data, null, 2)}`;
      if (options.input?.validate === true) {
        const vResult = await httpGet(`${apiBase}/config/validate`, apiKey);
        const vData = parseJsonSafe(vResult.body);
        text += vResult.status === 200
          ? `\n\nValidation:\n${formatValidation(vData)}`
          : `\n\nValidation error (${vResult.status}): ${formatDetail(vData, vResult.body)}`;
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(text)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),
      ]);
    }
  }

}

/**
 * set_config — one batched POST /config/set {settings, persist}. The server
 * resolves every key first, so an unknown key rejects the whole batch and
 * nothing is applied. persist defaults to false (running server only).
 */
export class AxonConfigSetTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    const keys = Object.keys(options.input?.settings || {}).join(', ');
    return {
      invocationMessage: `Applying Axon config changes: ${keys}...`,
    };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = resolveApiBase();
    const apiKey = config.get<string>('apiKey', '');
    const settings: Record<string, any> = options.input?.settings || {};
    const persist: boolean = options.input?.persist === true; // default false
    if (Object.keys(settings).length === 0) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart('No changes provided.'),
      ]);
    }
    try {
      const result = await httpPost(`${apiBase}/config/set`, { settings, persist }, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(`Config not changed (${result.status}): ${formatDetail(data, result.body)}`),
        ]);
      }
      const applied: any[] = Array.isArray(data.applied) ? data.applied : [];
      const lines = [`Applied ${applied.length} config change(s)${persist ? ' (saved to config.yaml)' : ' (running server only)'}:`];
      for (const a of applied) {
        lines.push(`  ✓ ${a.key} = ${JSON.stringify(a.new_value)} (was ${JSON.stringify(a.old_value)})`);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(lines.join('\n')),
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),
      ]);
    }
  }

}

// ---------------------------------------------------------------------------

// VS Code command: axon.configSetup — multi-step QuickPick wizard

// ---------------------------------------------------------------------------

export async function runConfigSetupWizard(apiBase: string, apiKey: string): Promise<void> {
  const changes: Record<string, any> = {};
  // Step 1: LLM provider
  const providerPick = await vscode.window.showQuickPick(
    ['ollama', 'openai', 'gemini', 'grok', 'vllm', 'copilot'],
    { title: 'Axon Config Setup (1/5)', placeHolder: 'Select LLM provider' }
  );
  if (providerPick === undefined) {
    return; // cancelled
  }
  changes['llm.provider'] = providerPick;
  // Step 2: LLM model
  const modelInput = await vscode.window.showInputBox({
    title: 'Axon Config Setup (2/5)',
    prompt: 'LLM model name',
    placeHolder: providerPick === 'ollama' ? 'llama3.1:8b' : providerPick === 'openai' ? 'gpt-4o' : 'gemini-2.0-flash',
  });
  if (modelInput === undefined) {
    return;
  }
  if (modelInput.trim()) {
    changes['llm.model'] = modelInput.trim();
  }
  // Step 3: Embedding provider
  const embedPick = await vscode.window.showQuickPick(
    ['sentence_transformers', 'ollama', 'fastembed', 'openai'],
    { title: 'Axon Config Setup (3/5)', placeHolder: 'Select embedding provider' }
  );
  if (embedPick === undefined) {
    return;
  }
  changes['embedding.provider'] = embedPick;
  // Step 4: Chunk strategy
  const chunkPick = await vscode.window.showQuickPick(
    [
      { label: 'recursive', description: 'Fast, text-based recursive splitting (default for code)' },
      { label: 'semantic', description: 'Sentence-aware semantic splitting' },
      { label: 'markdown', description: 'Header-aware Markdown splitting' },
      { label: 'cosine_semantic', description: 'Cosine-similarity-based sentence grouping' },
    ],
    { title: 'Axon Config Setup (4/5)', placeHolder: 'Select chunking strategy' }
  );
  if (chunkPick === undefined) {
    return;
  }
  changes['chunk.strategy'] = chunkPick.label;
  // Step 5: RAG toggles (multi-select)
  const togglePick = await vscode.window.showQuickPick(
    [
      { label: 'hybrid_search', description: 'BM25 + vector hybrid search', picked: true },
      { label: 'rerank', description: 'Cross-encoder re-ranking', picked: false },
      { label: 'sentence_window', description: 'Sentence-window context expansion', picked: false },
    ],
    {
      title: 'Axon Config Setup (5/5)',
      placeHolder: 'Select RAG features to enable',
      canPickMany: true,
    }
  );
  if (togglePick === undefined) {
    return;
  }
  const enabledToggles = new Set(togglePick.map((p: any) => p.label));
  changes['rag.hybrid_search'] = enabledToggles.has('hybrid_search');
  changes['rag.rerank'] = enabledToggles.has('rerank');
  changes['rag.sentence_window'] = enabledToggles.has('sentence_window');
  // Summary + confirm
  const summaryLines = Object.entries(changes).map(([k, v]) => `${k} = ${JSON.stringify(v)}`);
  const confirmPick = await vscode.window.showQuickPick(
    ['Apply and save', 'Cancel'],
    {
      title: 'Confirm config changes',
      placeHolder: summaryLines.join('  |  '),
    }
  );
  if (!confirmPick || confirmPick === 'Cancel') {
    vscode.window.showInformationMessage('Config setup cancelled.');
    return;
  }
  // Apply changes
  let applied = 0;
  let failed = 0;
  for (const [key, value] of Object.entries(changes)) {
    try {
      const result = await httpPost(`${apiBase}/config/set`, { key, value, persist: true }, apiKey);
      if (result.status === 200) {
        applied++;
      } else {
        failed++;
        const isSensitiveKey = /key|secret|password|token/i.test(key);
        state.outputChannel.appendLine(`Config set failed for ${key}: HTTP ${result.status}${isSensitiveKey ? '' : ` — ${result.body}`}`);
      }
    } catch (err) {
      failed++;
      state.outputChannel.appendLine(`Config set error for ${key}: ${err}`);
    }
  }
  if (failed === 0) {
    vscode.window.showInformationMessage(`Axon config updated: ${applied} setting(s) saved.`);
  } else {
    vscode.window.showWarningMessage(
      `Axon config: ${applied} saved, ${failed} failed. See Axon output channel for details.`
    );
  }

}

