/**
 * Sealed-store security LM tools and VS Code command implementations.
 */
import * as vscode from 'vscode';

import { state } from '../shared';
import { httpGet, httpPost, formatDetail, parseJsonSafe, apiConnectionError } from '../client/http';

function formatSecurityStatus(data: any): string {
  const initialized = Boolean(data?.initialized);
  const unlocked = Boolean(data?.unlocked);
  const lines = [
    `initialized: ${initialized}`,
    `unlocked: ${unlocked}`,
    `locked: ${initialized && !unlocked}`,
    `sealed_hidden_count: ${Number(data?.sealed_hidden_count || 0)}`,
  ];
  if (data?.public_key_fingerprint) {
    lines.push(`public_key_fingerprint: ${data.public_key_fingerprint}`);
  }
  if (data?.cipher_suite) {
    lines.push(`cipher_suite: ${data.cipher_suite}`);
  }
  return lines.join('\n');
}

export class AxonSecurityStatusTool implements vscode.LanguageModelTool<any> {
  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpGet(`${apiBase}/security/status`, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(
            `Security status error (${result.status}): ${formatDetail(data, result.body)}`,
          ),
        ]);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Axon security status:\n${formatSecurityStatus(data)}`),
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),
      ]);
    }
  }

}

export class AxonSecurityBootstrapTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Bootstrapping Axon sealed-store security...' };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { passphrase } = options.input ?? {};
    try {
      const result = await httpPost(`${apiBase}/security/bootstrap`, { passphrase }, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(
            `Security bootstrap failed: ${formatDetail(data, result.body)}`,
          ),
        ]);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Security bootstrapped.\n${formatSecurityStatus(data)}`),
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),
      ]);
    }
  }

}

export class AxonSecurityUnlockTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Unlocking Axon sealed-store security...' };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { passphrase } = options.input ?? {};
    try {
      const result = await httpPost(`${apiBase}/security/unlock`, { passphrase }, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(
            `Security unlock failed: ${formatDetail(data, result.body)}`,
          ),
        ]);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Security unlocked.\n${formatSecurityStatus(data)}`),
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),
      ]);
    }
  }

}

export class AxonSecurityLockTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Locking Axon sealed-store security...' };
  }
  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpPost(`${apiBase}/security/lock`, {}, apiKey);
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(
            `Security lock failed: ${formatDetail(data, result.body)}`,
          ),
        ]);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(`Security locked.\n${formatSecurityStatus(data)}`),
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),
      ]);
    }
  }

}

export class AxonSecurityChangePassphraseTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: 'Changing Axon sealed-store passphrase...' };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { old_passphrase, new_passphrase } = options.input ?? {};
    try {
      const result = await httpPost(
        `${apiBase}/security/change-passphrase`,
        { old_passphrase, new_passphrase },
        apiKey,
      );
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([
          new (vscode as any).LanguageModelTextPart(
            `Passphrase change failed: ${formatDetail(data, result.body)}`,
          ),
        ]);
      }
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(
          `Security passphrase changed.\n${formatSecurityStatus(data)}`,
        ),
      ]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([
        new (vscode as any).LanguageModelTextPart(apiConnectionError(err)),
      ]);
    }
  }

}

export async function showSecurityStatus(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    const result = await httpGet(`${apiBase}/security/status`, apiKey);
    const data = parseJsonSafe(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Security status failed — ${formatDetail(data, result.body)}`);
      return;
    }
    state.outputChannel.show();
    state.outputChannel.appendLine(`\n=== Axon Security Status ===\n${formatSecurityStatus(data)}\n`);
    vscode.window.showInformationMessage('Axon: Security status written to the Axon output channel.');
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to read security status. ${apiConnectionError(err)}`);
  }
}

export async function bootstrapSecurity(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const passphrase = await vscode.window.showInputBox({
    prompt: 'Enter a passphrase to bootstrap sealed-store security',
    password: true,
    ignoreFocusOut: true,
  });
  if (!passphrase) { return; }
  try {
    const result = await httpPost(`${apiBase}/security/bootstrap`, { passphrase }, apiKey);
    const data = parseJsonSafe(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Security bootstrap failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage('Axon: Sealed-store security bootstrapped.');
    state.outputChannel.appendLine(`Security bootstrapped. ${formatSecurityStatus(data)}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to bootstrap security. ${apiConnectionError(err)}`);
  }
}

export async function unlockSecurity(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const passphrase = await vscode.window.showInputBox({
    prompt: 'Enter the sealed-store passphrase to unlock Axon',
    password: true,
    ignoreFocusOut: true,
  });
  if (!passphrase) { return; }
  try {
    const result = await httpPost(`${apiBase}/security/unlock`, { passphrase }, apiKey);
    const data = parseJsonSafe(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Security unlock failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage('Axon: Sealed-store unlocked.');
    state.outputChannel.appendLine(`Security unlocked. ${formatSecurityStatus(data)}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to unlock security. ${apiConnectionError(err)}`);
  }
}

export async function lockSecurity(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  try {
    const result = await httpPost(`${apiBase}/security/lock`, {}, apiKey);
    const data = parseJsonSafe(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Security lock failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage('Axon: Sealed-store locked.');
    state.outputChannel.appendLine(`Security locked. ${formatSecurityStatus(data)}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to lock security. ${apiConnectionError(err)}`);
  }
}

export async function changeSecurityPassphrase(apiBase: string): Promise<void> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiKey = config.get<string>('apiKey', '');
  const oldPassphrase = await vscode.window.showInputBox({
    prompt: 'Enter the current sealed-store passphrase',
    password: true,
    ignoreFocusOut: true,
  });
  if (!oldPassphrase) { return; }
  const newPassphrase = await vscode.window.showInputBox({
    prompt: 'Enter the new sealed-store passphrase',
    password: true,
    ignoreFocusOut: true,
  });
  if (!newPassphrase) { return; }
  try {
    const result = await httpPost(
      `${apiBase}/security/change-passphrase`,
      { old_passphrase: oldPassphrase, new_passphrase: newPassphrase },
      apiKey,
    );
    const data = parseJsonSafe(result.body);
    if (result.status !== 200) {
      vscode.window.showErrorMessage(`Axon: Passphrase change failed — ${formatDetail(data, result.body)}`);
      return;
    }
    vscode.window.showInformationMessage('Axon: Sealed-store passphrase updated.');
    state.outputChannel.appendLine(`Security passphrase changed. ${formatSecurityStatus(data)}`);
  } catch (err) {
    vscode.window.showErrorMessage(`Axon: Failed to change passphrase. ${apiConnectionError(err)}`);
  }
}

// ---------------------------------------------------------------------------
// seal_project LM tool (SP-B1 parity sweep)
// ---------------------------------------------------------------------------

export class AxonSealProjectTool implements vscode.LanguageModelTool<any> {
  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {
    return { invocationMessage: `Sealing project "${options.input?.project_name}"…` };
  }
  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    const { project_name, migration_mode } = options.input ?? {};
    try {
      const result = await httpPost(
        `${apiBase}/project/seal`,
        { project_name, migration_mode: migration_mode ?? 'in_place' },
        apiKey,
      );
      const data = parseJsonSafe(result.body);
      if (result.status !== 200) {
        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(
          `Seal failed: ${formatDetail(data, result.body)}`
        )]);
      }
      const status = data.status ?? 'sealed';
      const files = data.files_sealed ?? 0;
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(
        `Project '${project_name}': ${status} (${files} files encrypted at rest).`
      )]);
    } catch (err) {
      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);
    }
  }
}
