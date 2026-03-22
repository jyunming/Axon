/**
 * AxonStore share LM tools and VS Code command implementations.
 */
import * as vscode from 'vscode';

import { state } from '../shared';
import { httpGet, httpPost, formatDetail, parseJsonSafe, apiConnectionError } from '../client/http';

// ---------------------------------------------------------------------------
// LM Tools
// ---------------------------------------------------------------------------

export class AxonShareProjectTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    const { project, grantee } = options.input ?? {};
    return { invocationMessage: `Generating share key for project "${project}" → ${grantee}…` };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { project, grantee } = options.input ?? {};
    try {
      const result = await httpPost(`${apiBase}/share/generate`, { project, grantee }, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Share generation failed: ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(
        `Share key generated.\nProject: ${data.project}\nGrantee: ${data.grantee}\nAccess: read-only\nKey ID: ${data.key_id}\n\nShare string (send to ${data.grantee}):\n${data.share_string}`
      )]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}

export class AxonRedeemShareTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Redeeming share key and mounting shared project…' };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { share_string } = options.input ?? {};
    try {
      const result = await httpPost(`${apiBase}/share/redeem`, { share_string }, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Redeem failed: ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(
        `Share redeemed. Project "${data.owner}/${data.project}" mounted as "${data.mount_name}" (read-only).`
      )]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}

export class AxonRevokeShareTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: `Revoking share key ${options.input?.key_id}…` };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { key_id } = options.input ?? {};
    try {
      const result = await httpPost(`${apiBase}/share/revoke`, { key_id }, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Revoke failed: ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(
        `Share ${data.key_id} revoked. Grantee "${data.grantee}" no longer has access to "${data.project}".`
      )]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}

export class AxonListSharesTool implements vscode.LanguageModelTool<any> {
  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpGet(`${apiBase}/share/list`, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Share list error: ${formatDetail(data, result.body)}`)]);
      }
      const text = JSON.stringify(data, null, 2);
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(text)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}

export class AxonInitStoreTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: `Initialising AxonStore at "${options.input.base_path}"...` };
  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { base_path } = options.input;
    try {
      const result = await httpPost(`${apiBase}/store/init`, { base_path }, apiKey);
      const data = JSON.parse(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`AxonStore init error (${result.status}): ${formatDetail(data, result.body)}`)]);
      }
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`AxonStore initialised at ${data.store_path} (user: ${data.username}). Share tools are now available.`)]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}

// ---------------------------------------------------------------------------
// VS Code command implementations
// ---------------------------------------------------------------------------

export async function initStore(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  let basePath = config.get<string>('storeBase', '');
  if (!basePath) {
    basePath = await vscode.window.showInputBox({
      prompt: 'Enter the base path for AxonStore (e.g. /data or ~/axon-data)',
      placeHolder: '/data',
    }) || '';
  }
  if (!basePath) { return; }
  try {
    const result = await httpPost(`${apiBase}/store/init`, { base_path: basePath }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Store init failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: AxonStore initialized at ${data.store_path} (user: ${data.username})`
    );
    state.outputChannel.appendLine(`AxonStore: ${data.store_path}, user dir: ${data.user_dir}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to initialize store. ${apiConnectionError(err)}`);
  }
}

export async function shareProject(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const project = await vscode.window.showInputBox({ prompt: 'Project name to share', placeHolder: 'research' });
  if (!project) { return; }
  const grantee = await vscode.window.showInputBox({ prompt: 'Grantee username (OS username on shared filesystem)', placeHolder: 'bob' });
  if (!grantee) { return; }
  try {
    const result = await httpPost(`${apiBase}/share/generate`, { project, grantee }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Share generation failed — ${formatDetail(data, result.body)}`);
      return;
    }
    await vscode.env.clipboard.writeText(data.share_string);
    vscode.window.showInformationMessage(
      `Axon: Share key copied to clipboard (key: ${data.key_id}). Send the share string to ${grantee}.`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to generate share key. ${apiConnectionError(err)}`);
  }
}

export async function redeemShare(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const share_string = await vscode.window.showInputBox({
    prompt: 'Paste the share string you received from the project owner',
    placeHolder: 'base64 share string...',
  });
  if (!share_string) { return; }
  try {
    const result = await httpPost(`${apiBase}/share/redeem`, { share_string }, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Redeem failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage(
      `Axon: Mounted "${data.owner}/${data.project}" as "${data.mount_name}" (read-only)`
    );
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to redeem share. ${apiConnectionError(err)}`);
  }
}

export async function revokeShare(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    const listResult = await httpGet(`${apiBase}/share/list`, apiKey);
    const data = JSON.parse(listResult.body);
    const active = (data.sharing || []).filter((s: any) => !s.revoked);
    if (active.length === 0) {
      vscode.window.showInformationMessage('Axon: No active shares to revoke.');
      return;
    }
    const items = active.map((s: any) => ({
      label: `${s.project} → ${s.grantee}`,
      description: `key: ${s.key_id} | read-only`,
      key_id: s.key_id,
    }));
    const picked = await vscode.window.showQuickPick(items, { placeHolder: 'Select share to revoke' });
    if (!picked) { return; }
    const revokeResult = await httpPost(`${apiBase}/share/revoke`, { key_id: (picked as any).key_id }, apiKey);
    const revokeData = JSON.parse(revokeResult.body);
    if (revokeResult.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Revoke failed — ${revokeData.detail || revokeResult.body}`);
      return;
    }
    vscode.window.showInformationMessage(`Axon: Revoked access for ${revokeData.grantee} to ${revokeData.project}.`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to revoke share. ${apiConnectionError(err)}`);
  }
}

export async function listShares(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    const result = await httpGet(`${apiBase}/share/list`, apiKey);
    const data = JSON.parse(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: List shares failed.`);
      return;
    }
    const sharing = (data.sharing || []).map((s: any) =>
      `  • ${s.project} → ${s.grantee} [ro]${s.revoked ? ' (revoked)' : ''}`
    ).join('\n') || '  (none)';
    const shared = (data.shared || []).map((s: any) =>
      `  • ${s.owner}/${s.project} mounted as ${s.mount} [ro]`
    ).join('\n') || '  (none)';
    state.outputChannel.show();
    state.outputChannel.appendLine(`\n=== Axon Shares ===\nSharing with others:\n${sharing}\n\nShared with me:\n${shared}\n`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to list shares. ${apiConnectionError(err)}`);
  }
}
