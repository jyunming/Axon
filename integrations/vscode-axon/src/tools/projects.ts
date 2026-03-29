/**

 * Project management LM tools and VS Code command implementations.

 */

import * as vscode from 'vscode';

import { state } from '../shared';

import { httpGet, httpPost, formatDetail, apiConnectionError } from '../client/http';

// ---------------------------------------------------------------------------

// LM Tools

// ---------------------------------------------------------------------------

export class AxonListProjectsTool implements vscode.LanguageModelTool<any> {

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    try {

      const result = await httpGet(`${apiBase}/projects`, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      const names = (data.projects || []).map((p: any) => p.name).join(', ');

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Projects: ${names || 'None'}`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonSwitchProjectTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    return {

      invocationMessage: `Switching Axon active project to: "${options.input.name}"...`

    };

  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    const { name } = options.input;

    try {

      const result = await httpPost(`${apiBase}/project/switch`, { name }, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Switched to project: ${data.active_project || name}`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonCreateProjectTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    return {

      invocationMessage: `Creating Axon project: "${options.input.name}"...`

    };

  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    const { name, description = '' } = options.input;

    try {

      const result = await httpPost(`${apiBase}/project/new`, { name, description }, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Project Creation Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Project: ${data.project}`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonDeleteProjectTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    return {

      invocationMessage: `Deleting Axon project: "${options.input.name}"...`

    };

  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    const { name } = options.input;

    try {

      const result = await httpPost(`${apiBase}/project/delete/${name}`, {}, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Project Deletion Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Message: ${data.message}`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonDeleteDocumentsTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    return {

      invocationMessage: `Deleting ${options.input.docIds.length} documents from Axon...`

    };

  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    const { docIds } = options.input;

    try {

      const result = await httpPost(`${apiBase}/delete`, { doc_ids: docIds }, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Document Deletion Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Deleted: ${data.deleted}`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonListKnowledgeTool implements vscode.LanguageModelTool<any> {

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    try {

      const result = await httpGet(`${apiBase}/collection`, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      const files = (data.files || []).map((f: any) => `${f.source} (${f.chunks} chunks)`).join('\n');

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Total Files: ${data.total_files}\nTotal Chunks: ${data.total_chunks}\n\nFiles:\n${files}`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonUpdateSettingsTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(_options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    return {

      invocationMessage: `Updating Axon RAG settings...`

    };

  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    try {

      const result = await httpPost(`${apiBase}/config/update`, options.input, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon Configuration Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Status: ${data.status}, Settings Applied.`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonGetCurrentSettingsTool implements vscode.LanguageModelTool<any> {

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    try {

      const result = await httpGet(`${apiBase}/config`, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Axon API Error (${result.status}): ${formatDetail(data, result.body)}`)]);

      }

      const summary = JSON.stringify(data, null, 2);

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Current Axon settings:\n${summary}`)]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonListSessionsTool implements vscode.LanguageModelTool<any> {

  async invoke(_options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    try {

      const result = await httpGet(`${apiBase}/sessions`, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Sessions error: ${formatDetail(data, result.body)}`)]);

      }

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(JSON.stringify(data, null, 2))]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

export class AxonGetSessionTool implements vscode.LanguageModelTool<any> {

  async prepareInvocation(options: vscode.LanguageModelToolInvocationPrepareOptions<any>, _token: vscode.CancellationToken) {

    return { invocationMessage: `Loading session ${options.input?.session_id}…` };

  }

  async invoke(options: vscode.LanguageModelToolInvocationOptions<any>, _token: vscode.CancellationToken) {

    const config = vscode.workspace.getConfiguration('axon');

    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');

    const apiKey = config.get<string>('apiKey', '');

    const { session_id } = options.input ?? {};

    try {

      const result = await httpGet(`${apiBase}/session/${encodeURIComponent(session_id)}`, apiKey);

      const data = JSON.parse(result.body);

      if (result.status !== 200) {

        return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(`Session error: ${formatDetail(data, result.body)}`)]);

      }

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(JSON.stringify(data, null, 2))]);

    } catch (err) {

      return new (vscode as any).LanguageModelToolResult([new (vscode as any).LanguageModelTextPart(apiConnectionError(err))]);

    }

  }

}

// ---------------------------------------------------------------------------

// VS Code command implementations

// ---------------------------------------------------------------------------

export async function switchProject(apiBase: string): Promise<void> {

  const config = vscode.workspace.getConfiguration('axon');

  const apiKey = config.get<string>('apiKey', '');

  let allNames: string[] = [];

  try {

    const result = await httpGet(`${apiBase}/projects`, apiKey);

    const data = JSON.parse(result.body);

    const local: string[] = (data.projects ?? []).map((p: any) => p.name);

    const mounts: string[] = (data.shared_mounts ?? []).map((m: any) => m.name);

    allNames = [...local, ...mounts];

  } catch (err) {

    vscode.window.showErrorMessage(`Axon: Failed to list projects. ${apiConnectionError(err)}`);

    return;

  }

  if (allNames.length === 0) {

    vscode.window.showInformationMessage('Axon: No projects found.');

    return;

  }

  const selected = await vscode.window.showQuickPick(allNames, {

    placeHolder: 'Select an Axon project (including mounts)',

  });

  if (!selected) {

    return;

  }

  try {

    await httpPost(`${apiBase}/project/switch`, { name: selected }, apiKey);

    vscode.window.showInformationMessage(`Axon: Switched to project "${selected}".`);

    state.outputChannel.appendLine(`Switched to project: ${selected}`);

  } catch (err) {

    vscode.window.showErrorMessage(`Axon: Failed to switch project. ${apiConnectionError(err)}`);

  }

}

export async function createNewProject(apiBase: string): Promise<void> {

  const config = vscode.workspace.getConfiguration('axon');

  const apiKey = config.get<string>('apiKey', '');

  const name = await vscode.window.showInputBox({

    prompt: 'Enter a name for the new Axon project',

    placeHolder: 'e.g. project-alpha'

  });

  if (!name) {

    return;

  }

  const description = await vscode.window.showInputBox({

    prompt: 'Optional: Enter a description',

    placeHolder: 'Documentation for system architecture...'

  });

  try {

    await httpPost(`${apiBase}/project/new`, { name, description: description || '' }, apiKey);

    vscode.window.showInformationMessage(`Axon: Created project "${name}".`);

    state.outputChannel.appendLine(`Created project: ${name}`);

    // Auto-switch to it

    await httpPost(`${apiBase}/project/switch`, { name }, apiKey);

  } catch (err) {

    vscode.window.showErrorMessage(`Axon: Failed to create project. ${apiConnectionError(err)}`);

  }

}

