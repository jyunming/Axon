/**

 * Config validation and wizard LM tools + VS Code command implementations.

 */

import * as vscode from 'vscode';

import { state } from '../shared';

import { httpGet, httpPost, formatDetail, apiConnectionError } from '../client/http';

// ---------------------------------------------------------------------------

// LM Tools

// ---------------------------------------------------------------------------

export class AxonConfigValidateTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    return {

      invocationMessage: 'Validating Axon config.yaml...',

    };

  }

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    try {

      const result = await httpGet(`${apiBase}/config/validate`, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([

          new (vscode as any).LanguageModelTextPart(

            `Axon API Error (${result.status}): ${formatDetail(data, result.body)}`

          ),

        ]);

      }

      const issues: any[] = data.issues || [];

      const valid: boolean = data.valid ?? true;

      if (issues.length === 0) {

        return new (vscode as any).LanguageModelToolResult([

          new (vscode as any).LanguageModelTextPart('Config validation passed. No issues found.'),

        ]);

      }

      const lines = issues.map((issue: any) => {

        const suggestion = issue.suggestion ? ` Suggestion: ${issue.suggestion}` : '';

        return `[${issue.level.toUpperCase()}] ${issue.section}.${issue.field}: ${issue.message}${suggestion}`;

      });

      const summary = valid

        ? `Config has ${issues.length} notice(s) (no errors):`

        : `Config has errors (${issues.filter((i: any) => i.level === 'error').length} error(s)):`;

      return new (vscode as any).LanguageModelToolResult([

        new (vscode as any).LanguageModelTextPart(`${summary}\n${lines.join('\n')}`),

      ]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([

        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),

      ]);

    }

  }

}

export class AxonConfigSetTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    const changes = options.input.changes || {};

    const keys = Object.keys(changes).join(', ');

    return {

      invocationMessage: `Applying Axon config changes: ${keys}...`,

    };

  }

  /**

   * Accepts input of the form: { changes: { "chunk.strategy": "markdown", "rag.top_k": 15 } }

   * Each key is a dot-notation config field; calls POST /config/set for each change.

   */

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    const changes: Record<string, any> = options.input.changes || {};

    const persist: boolean = options.input.persist !== false; // default true

    const results: string[] = [];

    const errors: string[] = [];

    for (const [key, value] of Object.entries(changes)) {

      try {

        const result = await httpPost(`${apiBase}/config/set`, { key, value, persist }, apiKey);

        const data = JSON.parse(result.body);

        if (result.status !== 200) {

          errors.push(`${key}: API Error (${result.status}) — ${formatDetail(data, result.body)}`);

        } else {

          results.push(`${key} = ${JSON.stringify(data.new_value)} (was ${JSON.stringify(data.old_value)})`);

        }

      } catch (err) {

        errors.push(`${key}: ${err instanceof Error ? err.message : String(err)}`);

      }

    }

    const lines: string[] = [];

    if (results.length > 0) {

      lines.push(`Applied ${results.length} config change(s):`);

      results.forEach(r => lines.push(`  ✓ ${r}`));

    }

    if (errors.length > 0) {

      lines.push(`${errors.length} error(s):`);

      errors.forEach(e => lines.push(`  ✗ ${e}`));

    }

    if (lines.length === 0) {

      lines.push('No changes provided.');

    }

    return new (vscode as any).LanguageModelToolResult([

      new (vscode as any).LanguageModelTextPart(lines.join('\n')),

    ]);

  }

}

// ---------------------------------------------------------------------------

// VS Code command: axon.configSetup — multi-step QuickPick wizard

// ---------------------------------------------------------------------------

export async function runConfigSetupWizard(apiBase: string, apiKey: string): Promise<void> {

  const changes: Record<string, any> = {};

  // Step 1: LLM provider

  const providerPick = await vscode.window.showQuickPick(

    ['ollama', 'openai', 'gemini', 'grok', 'vllm', 'github_copilot'],

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

        state.outputChannel.appendLine(`Config set failed for ${key}: ${result.body}`);

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

