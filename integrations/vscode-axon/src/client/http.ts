/**
 * HTTP helpers — no external dependencies, uses built-in Node.js http/https.
 */
import * as http from 'http';
import * as https from 'https';

export interface HttpResult {
  status: number;
  body: string;
}

export interface SearchChunk {
  id: string;
  text: string;
  score: number;
  metadata: Record<string, string> | null;
}

/** Safely serialize a FastAPI error detail, which may be a string, array, or object. */
export function formatDetail(data: any, fallback: string): string {
  const d = data?.detail;
  if (d === undefined || d === null) { return fallback; }
  if (typeof d === 'string') { return d; }
  return JSON.stringify(d);
}

export function parseJsonSafe(body: string): any {
  try {
    return JSON.parse(body);
  } catch {
    return {};
  }
}

export function normalizeGraphPayload(data: any): { nodes: any[]; links: any[] } {
  if (data && Array.isArray(data.nodes) && Array.isArray(data.links)) {
    return data;
  }
  return { nodes: [], links: [] };
}

export function httpGet(url: string, apiKey?: string, timeoutMs: number = 5000, surface: string = 'vscode_extension_tool'): Promise<HttpResult> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const lib = parsed.protocol === 'https:' ? https : http;
    const headers: Record<string, string> = { 'Content-Type': 'application/json', 'X-Axon-Surface': surface };
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }
    const req = lib.get({ hostname: parsed.hostname, port: parsed.port, path: parsed.pathname + parsed.search, headers }, (res) => {
      let body = '';
      res.on('data', (chunk: Buffer) => { body += chunk.toString(); });
      res.on('end', () => resolve({ status: res.statusCode ?? 0, body }));
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`));
    });
  });
}

export function httpPost(url: string, payload: unknown, apiKey?: string, timeoutMs: number = 20_000, surface: string = 'vscode_extension_tool'): Promise<HttpResult> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const lib = parsed.protocol === 'https:' ? https : http;
    const body = JSON.stringify(payload);
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body).toString(),
      'X-Axon-Surface': surface,
    };
    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }
    const req = lib.request(
      { method: 'POST', hostname: parsed.hostname, port: parsed.port, path: parsed.pathname + parsed.search, headers },
      (res) => {
        let resBody = '';
        res.on('data', (chunk: Buffer) => { resBody += chunk.toString(); });
        res.on('end', () => resolve({ status: res.statusCode ?? 0, body: resBody }));
      },
    );
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy();
      reject(new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`));
    });
    req.write(body);
    req.end();
  });
}

export async function searchAxon(
  apiBase: string,
  apiKey: string,
  query: string,
  topK: number,
  threshold?: number,
  filters?: Record<string, any>,
  project?: string,
  surface: string = 'vscode_extension_tool',
): Promise<SearchChunk[]> {
  const body: any = { query, top_k: topK };
  if (threshold != null) { body.threshold = threshold; }
  if (filters != null) { body.filters = filters; }
  if (project != null) { body.project = project; }
  const result = await httpPost(`${apiBase}/search`, body, apiKey || undefined, 20_000, surface);
  if (result.status !== 200) {
    throw new Error(`Search returned HTTP ${result.status}: ${result.body}`);
  }
  const data = JSON.parse(result.body);
  // /search returns a plain array, not { results: [...] }
  return Array.isArray(data) ? data : (data.results ?? []);
}

/** Produce a consistent connection-error message for tool catch blocks. */
export function apiConnectionError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  return `Could not reach Axon API: ${msg}. Run \`axon-api\` in a terminal, or enable the \`axon.autoStart\` setting in VS Code.`;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
