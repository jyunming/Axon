/**
 * Axon API Client
 * Wraps FastAPI endpoints for the WebGUI.
 */

class AxonAPI {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.activeProject = 'default';
        this.apiKey = localStorage.getItem('axon_api_key') || '';
    }

    setApiKey(key) {
        this.apiKey = key;
        localStorage.setItem('axon_api_key', key);
    }

    setProject(projectName) {
        this.activeProject = projectName;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Content-Type': 'application/json',
            'X-Axon-Surface': 'webgui',
            ...options.headers
        };

        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }

        const response = await fetch(url, {
            ...options,
            headers
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'API Request Failed');
        }

        return response.json();
    }

    // Projects
    async listProjects() {
        const res = await this.request('/projects');
        return res.projects || [];
    }

    async switchProject(projectName) {
        const res = await this.request('/project/switch', {
            method: 'POST',
            body: JSON.stringify({ project_name: projectName })
        });
        this.activeProject = projectName;
        return res;
    }

    // Query & Search
    async query(question, overrides = {}) {
        const payload = {
            query: question,
            project: this.activeProject,
            ...overrides
        };

        return this.request('/query', {
            method: 'POST',
            body: JSON.stringify(payload)
        });
    }

    /**
     * Stream a query response
     * @param {string} question
     * @param {object} overrides
     * @param {function} onToken Callback for each text token
     * @param {function} onDiagnostic Callback for diagnostic steps
     */
    async streamQuery(question, overrides = {}, onToken, onDiagnostic) {
        const payload = {
            query: question,
            project: this.activeProject,
            stream: true,
            ...overrides
        };

        const headers = {
            'Content-Type': 'application/json',
            'X-Axon-Surface': 'webgui'
        };

        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }

        const response = await fetch(`${this.baseUrl}/query/stream`, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error('Streaming failed');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // FastAPI/SSE events are usually newline separated or JSON chunks
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.trim() || !line.startsWith('data: ')) continue;

                const content = line.substring(6).trim();
                if (!content) continue;

                try {
                    // Try to parse as JSON (for dict chunks like sources)
                    const data = JSON.parse(content);

                    if (data.type === 'sources') {
                        onToken({ type: 'sources', content: data.sources });
                    } else if (data.type === 'diagnostics') {
                        onDiagnostic(data.content);
                    } else if (data.type === 'error') {
                        throw new Error(data.content);
                    } else {
                        // Fallback: if it's JSON but not a known type, just send as is
                        onToken({ type: 'token', content: JSON.stringify(data) });
                    }
                } catch (e) {
                    // If not JSON, it's a raw text token
                    onToken({ type: 'token', content: content });
                }
            }
        }
    }

    // Graph
    async getGraphData() {
        return this.request('/graph/data', {
            params: { project: this.activeProject }
        });
    }

    // Ingest / Collection
    async getCollection() {
        return this.request('/collection');
    }

    async ingestPath(path) {
        return this.request('/ingest', {
            method: 'POST',
            body: JSON.stringify({ path })
        });
    }

    async ingestURL(url) {
        return this.request('/ingest/url', {
            method: 'POST',
            body: JSON.stringify({ url, project: this.activeProject })
        });
    }

    // Config
    async getConfig() {
        return this.request('/config');
    }

    async updateConfig(updates) {
        return this.request('/config/update', {
            method: 'POST',
            body: JSON.stringify({ ...updates, persist: true })
        });
    }

    // Governance
    async getAuditLogs(limit = 50) {
        return this.request(`/governance/audit?limit=${limit}`);
    }
}

export default new AxonAPI();
