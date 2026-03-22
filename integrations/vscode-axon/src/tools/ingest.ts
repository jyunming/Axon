/**
 * Ingest LM tools and VS Code command implementations.
 */
import * as vscode from 'vscode';
import * as path from 'path';

import { state } from '../shared';
import { httpGet, httpPost, formatDetail, parseJsonSafe } from '../client/http';

// ---------------------------------------------------------------------------
// LM Tools
// ---------------------------------------------------------------------------

export class AxonIngestTextTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { text, source = 'agent_input', project } = options.input;

    try {
      const body: any = { text, metadata: { source } };
      if (project != null) { body.project = project; }
      const result = await httpPost(`${apiBase}/add_text`, body, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Success: ${data.status}, ID: ${data.doc_id}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during text ingest: ${err}`)]);
    }
  }
}

export class AxonIngestUrlTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { url, project } = options.input;

    try {
      const body: any = { url };
      if (project != null) { body.project = project; }
      const result = await httpPost(`${apiBase}/ingest_url`, body, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon URL Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, URL: ${data.url}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during URL ingest: ${err}`)]);
    }
  }
}

export class AxonIngestPathTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return {
      invocationMessage: `Ingesting local path into Axon: "${options.input.path}"...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { path: ingestPath } = options.input;

    try {
      const result = await httpPost(`${apiBase}/ingest`, { path: ingestPath }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Path Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Message: ${data.message}, JobID: ${data.job_id}`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error during path ingest: ${err}`)]);
    }
  }
}

export class AxonGetIngestStatusTool implements vscode.LanguageModelTool<any> {
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { job_id } = options.input;

    try {
      const result = await httpGet(`${apiBase}/ingest/status/${encodeURIComponent(job_id)}`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status === 404) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Job not found: ${job_id}`)]);
      }
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error checking status (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      const status = data.status;
      if (status === 'completed') {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Ingestion complete. Status: completed. You can now search the ingested documents.`)]);
      } else if (status === 'failed') {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Ingestion failed. Error: ${data.error || 'unknown error'}`)]);
      } else {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Ingestion still in progress (status: ${status}). Wait a moment and check again.`)]);
      }
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Error checking ingest status: ${err}`)]);
    }
  }
}

/** Map file extension → MIME type for image ingest. */
function getImageMimeType(filePath: string): string {
  const ext = filePath.split('.').pop()?.toLowerCase() ?? '';
  const mimeMap: Record<string, string> = {
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    bmp: 'image/bmp',
    tif: 'image/tiff',
    tiff: 'image/tiff',
    webp: 'image/webp',
  };
  return mimeMap[ext] ?? 'image/png';
}

const SUPPORTED_IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp']);

export class AxonIngestImageTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return {
      invocationMessage: `Describing image "${options.input.imagePath}" with Copilot and ingesting into Axon...`
    };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, token: vscode.CancellationToken) {
    const { imagePath, project } = options.input as { imagePath: string; project?: string };
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');

    // Validate extension
    const ext = imagePath.split('.').pop()?.toLowerCase() ?? '';
    if (!SUPPORTED_IMAGE_EXTENSIONS.has(ext)) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(
          `Unsupported image format ".${ext}". Supported: PNG, JPG, JPEG, BMP, TIF, TIFF, WEBP.`
        )
      ]);
    }

    try {
      // Read the image file as Uint8Array (LanguageModelDataPart requires Uint8Array, not Buffer)
      const fs = await import('fs');
      const rawBuffer = fs.readFileSync(imagePath);
      const imageBuffer = new Uint8Array(rawBuffer.buffer, rawBuffer.byteOffset, rawBuffer.byteLength);
      const mimeType = getImageMimeType(imagePath);

      // Select a Copilot model with vision capability
      const models = await vscode.lm.selectChatModels({ vendor: 'copilot' });
      if (models.length === 0) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart('No Copilot language model available. Ensure GitHub Copilot is installed and signed in.')
        ]);
      }

      // Prefer a model that supports image input; fall back to first available
      const model = models.find((m: any) => m.capabilities?.supportsImageToText) ?? models[0];

      // Build the multimodal prompt
      const prompt = [
        vscode.LanguageModelChatMessage.User([
          new (vscode as any).LanguageModelDataPart(imageBuffer, mimeType),
          new vscode.LanguageModelTextPart(
            'Describe this image in detail for a searchable knowledge base. ' +
            'Include all visible text, diagram structure, key concepts, labels, ' +
            'relationships, and any quantitative data shown. Be thorough and precise.'
          )
        ])
      ];

      // Stream the description
      const response = await model.sendRequest(prompt, {}, token);
      let description = '';
      for await (const chunk of response.text) {
        description += chunk;
      }

      if (!description.trim()) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart('Copilot returned an empty description for the image.')
        ]);
      }

      // Ingest the description as text
      const body: Record<string, any> = {
        text: description,
        metadata: { type: 'image', original_path: imagePath, ingested_via: 'copilot_vision' }
      };
      if (project) {
        body['project'] = project;
      }
      const result = await httpPost(`${apiBase}/add_text`, body, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(`Axon Ingest Error (${result.status}): ${formatDetail(data, result.body)}`)
        ]);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(
          `Image ingested successfully. doc_id: ${data.doc_id}, status: ${data.status}. ` +
          `Description (${description.length} chars) stored in Axon.`
        )
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Error during image ingest: ${err}`)
      ]);
    }
  }
}

export class AxonRefreshIngestTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Re-ingesting changed files…' };
  }

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpPost(`${apiBase}/ingest/refresh`, {}, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Refresh error: ${formatDetail(data, result.body)}`)]);
      }
      const r = (data.reingested || []).length;
      const s = (data.skipped || []).length;
      const m = (data.missing || []).length;
      const e = (data.errors || []).length;
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(
        `Refresh complete: ${r} re-ingested, ${s} unchanged, ${m} missing, ${e} errors.\n` +
        (data.reingested?.length ? `Updated: ${data.reingested.join(', ')}` : '')
      )]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Refresh error: ${err}`)]);
    }
  }
}

export class AxonListStaleDocsTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    const days = options.input?.days ?? 7;
    return { invocationMessage: `Listing documents not refreshed in ${days} days…` };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const days = options.input?.days ?? 7;
    try {
      const result = await httpGet(`${apiBase}/collection/stale?days=${days}`, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Stale list error: ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(JSON.stringify(data, null, 2))]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Stale list error: ${err}`)]);
    }
  }
}

export class AxonClearKnowledgeBaseTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Clearing Axon knowledge base for current project…' };
  }

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpPost(`${apiBase}/clear`, {}, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Clear error: ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart('Knowledge base cleared for current project.')]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Clear error: ${err}`)]);
    }
  }
}

// ---------------------------------------------------------------------------
// VS Code command implementations
// ---------------------------------------------------------------------------

export async function ingestWorkspaceFolder(apiBase: string): Promise<void> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    vscode.window.showWarningMessage('Axon: No workspace folder is open.');
    return;
  }

  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const folderPath = workspaceFolders[0].uri.fsPath;

  try {
    const result = await httpPost(`${apiBase}/ingest`, { path: folderPath }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Ingest failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: Ingesting workspace folder "${folderPath}" (job ${data.job_id}). Check the Axon output panel for progress.`
    );
    state.outputChannel.appendLine(`Ingest workspace: ${folderPath} — job ${data.job_id}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to ingest workspace folder. Is the server running?`);
  }
}

export async function ingestPickedFolder(apiBase: string): Promise<void> {
  const selected = await vscode.window.showOpenDialog({
    canSelectFolders: true,
    canSelectFiles: true,
    canSelectMany: false,
    openLabel: 'Ingest into Axon',
    title: 'Select a folder or file to ingest into Axon',
  });

  if (!selected || selected.length === 0) {
    return;
  }

  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const selectedPath = selected[0].fsPath;

  try {
    const result = await httpPost(`${apiBase}/ingest`, { path: selectedPath }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Ingest failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: Ingesting "${selectedPath}" (job ${data.job_id}). Check the Axon output panel for progress.`
    );
    state.outputChannel.appendLine(`Ingest folder/file: ${selectedPath} — job ${data.job_id}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to ingest. Is the server running?`);
  }
}

export async function ingestCurrentFile(apiBase: string): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage('Axon: No active editor.');
    return;
  }

  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');

  const filePath = editor.document.uri.fsPath;
  const text = editor.document.getText();
  const source = path.basename(filePath);

  try {
    await httpPost(`${apiBase}/add_texts`, {
      docs: [{ text, metadata: { source: filePath } }],
    }, apiKey);
    vscode.window.showInformationMessage(`Axon: Ingested "${source}" into the knowledge base.`);
    state.outputChannel.appendLine(`Ingested file: ${filePath}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to ingest file. Is the server running?`);
  }
}

export async function refreshIngest(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    vscode.window.showInformationMessage('Axon: Checking for changed documents…');
    const result = await httpPost(`${apiBase}/ingest/refresh`, {}, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Refresh failed — ${formatDetail(data, result.body)}`);
      return;
    }
    const reingested = (data.reingested || []).length;
    const skipped = (data.skipped || []).length;
    const missing = (data.missing || []).length;
    const errors = (data.errors || []).length;
    state.outputChannel.show();
    state.outputChannel.appendLine(`\n=== Axon Refresh ===`);
    state.outputChannel.appendLine(`Re-ingested: ${reingested}  |  Unchanged: ${skipped}  |  Missing: ${missing}  |  Errors: ${errors}`);
    if (reingested > 0) {
      state.outputChannel.appendLine('Updated:');
      (data.reingested || []).forEach((s: string) => state.outputChannel.appendLine(`  ${s}`));
    }
    if (errors > 0) {
      state.outputChannel.appendLine('Errors:');
      (data.errors || []).forEach((e: any) => state.outputChannel.appendLine(`  ${e.source || e}: ${e.error || ''}`));
    }
    vscode.window.showInformationMessage(
      `Axon: Refresh complete — ${reingested} updated, ${skipped} unchanged${missing ? `, ${missing} missing` : ''}`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Refresh failed. Is the server running?`);
  }
}

export async function listStaleDocs(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const daysInput = await vscode.window.showInputBox({
    prompt: 'Show documents not refreshed in N days',
    value: '7',
    placeHolder: '7',
  });
  if (daysInput === undefined) { return; }
  const days = parseInt(daysInput, 10) || 7;
  try {
    const result = await httpGet(`${apiBase}/collection/stale?days=${days}`, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Stale list failed — ${formatDetail(data, result.body)}`);
      return;
    }
    const stale: any[] = data.stale_docs || [];
    state.outputChannel.show();
    state.outputChannel.appendLine(`\n=== Axon Stale Docs (>${days} days) ===`);
    if (stale.length === 0) {
      state.outputChannel.appendLine('All documents are fresh.');
      vscode.window.showInformationMessage(`Axon: All documents are fresh (threshold: ${days} days).`);
    } else {
      stale.sort((a, b) => b.age_days - a.age_days);
      stale.forEach(s => state.outputChannel.appendLine(`  ${String(s.age_days).padStart(5)}d  [${s.project}]  ${s.doc_id}`));
      state.outputChannel.appendLine(`Total: ${stale.length}`);
      vscode.window.showInformationMessage(`Axon: ${stale.length} stale document(s) found. Check Axon output panel.`);
    }
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to list stale documents.`);
  }
}

export async function clearKnowledgeBase(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const confirmed = await vscode.window.showWarningMessage(
    'Axon: Clear the entire knowledge base for the current project? This cannot be undone.',
    { modal: true },
    'Clear Knowledge Base',
  );
  if (confirmed !== 'Clear Knowledge Base') { return; }
  try {
    const result = await httpPost(`${apiBase}/clear`, {}, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Clear failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage('Axon: Knowledge base cleared for current project.');
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to clear knowledge base.`);
  }
}
