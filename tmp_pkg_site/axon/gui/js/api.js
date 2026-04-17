/**
 * Axon API Client
 * Wraps FastAPI endpoints for the WebGUI.
 */

class AxonAPI {
    constructor(baseUrl = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.activeProject = 'default';
        this.apiKey = localStorage.getItem('axon_api_key') || '';
        this.configKeyMap = {
            llm_provider: 'llm.provider',
            llm_model: 'llm.model',
            embedding_provider: 'embedding.provider',
            embedding_model: 'embedding.model',
            top_k: 'rag.top_k',
            similarity_threshold: 'rag.similarity_threshold',
            hybrid_search: 'rag.hybrid_search',
            hybrid_weight: 'rag.hybrid_weight',
            rerank: 'rerank.enabled',
            reranker_model: 'rerank.model',
            hyde: 'query_transformations.hyde',
            multi_query: 'query_transformations.multi_query',
            step_back: 'query_transformations.step_back',
            query_decompose: 'query_transformations.query_decompose',
            compress_context: 'context_compression.enabled',
            discussion_fallback: 'query_transformations.discussion_fallback',
            graph_rag: 'rag.graph_rag'
        };
    }

    setApiKey(key) {
        this.apiKey = key;
        localStorage.setItem('axon_api_key', key);
    }

    setProject(projectName) {
        this.activeProject = projectName;
    }

    async request(endpoint, options = {}) {
        const url = new URL(endpoint, this.baseUrl);
        const { params, headers: optionHeaders, ...fetchOptions } = options;
        const headers = {
            'X-Axon-Surface': 'webgui',
            ...optionHeaders
        };

        if (!(fetchOptions.body instanceof FormData) && !headers['Content-Type']) {
            headers['Content-Type'] = 'application/json';
        }

        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }

        if (params) {
            const searchParams = new URLSearchParams(url.search);
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined && value !== null && value !== '') {
                    searchParams.set(key, String(value));
                }
            });
            url.search = searchParams.toString();
        }

        const response = await fetch(url.toString(), {
            ...fetchOptions,
            headers
        });

        if (!response.ok) {
            const error = await response.json().catch(async () => ({
                detail: await response.text().catch(() => response.statusText)
            }));
            throw new Error(error.detail || 'API Request Failed');
        }

        return response.json();
    }

    // Projects
    async listProjects() {
        const res = await this.request('/projects');
        const projects = [];

        for (const group of [res.projects || [], res.memory_only || [], res.shared_mounts || []]) {
            for (const project of group) {
                if (project?.name) {
                    projects.push(project);
                }
            }
        }

        return projects;
    }

    async getHealth() {
        return this.request('/health');
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

        const response = await fetch(new URL('/query/stream', this.baseUrl).toString(), {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const error = await response.json().catch(async () => ({
                detail: await response.text().catch(() => 'Streaming failed')
            }));
            throw new Error(error.detail || 'Streaming failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        const processLine = (line) => {
            if (!line.trim() || !line.startsWith('data: ')) return;

            const content = line.substring(6).trim();
            if (!content) return;

            let data = null;

            try {
                data = JSON.parse(content);
            } catch (_error) {
                data = null;
            }

            if (data) {
                if (data.type === 'sources') {
                    onToken({ type: 'sources', content: data.sources });
                } else if (data.type === 'diagnostics') {
                    onDiagnostic(data.content);
                } else if (data.type === 'error') {
                    throw new Error(data.content || 'Streaming failed');
                } else if (data.type === 'token') {
                    onToken({ type: 'token', content: data.content || '' });
                } else {
                    onToken({ type: 'token', content: JSON.stringify(data) });
                }
                return;
            }

            if (content.startsWith('[ERROR]')) {
                throw new Error(content.replace(/^\[ERROR\]\s*/, ''));
            }

            onToken({ type: 'token', content: content });
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // FastAPI/SSE events are usually newline separated or JSON chunks
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                processLine(line);
            }
        }

        const trailing = buffer.trim();
        if (trailing) {
            processLine(trailing);
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
        return this.request('/ingest_url', {
            method: 'POST',
            body: JSON.stringify({ url, project: this.activeProject })
        });
    }

    async addText(text, metadata = {}, docId = null) {
        return this.request('/add_text', {
            method: 'POST',
            body: JSON.stringify({
                text,
                metadata,
                doc_id: docId,
                project: this.activeProject
            })
        });
    }

    async uploadFiles(files) {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file, file.name);
        });
        formData.append('project', this.activeProject);

        return this.request('/ingest/upload', {
            method: 'POST',
            body: formData
        });
    }

    // Config
    async getConfig() {
        return this.request('/config');
    }

    mapConfigKey(key) {
        return this.configKeyMap[key] || key;
    }

    async updateConfig(updates) {
        const entries = Object.entries(updates).filter(([, value]) => value !== undefined);
        if (entries.length === 0) {
            return { status: 'success', applied: [], results: [] };
        }

        const priority = new Map([
            ['llm_provider', 0],
            ['llm_model', 1],
            ['embedding_provider', 2],
            ['embedding_model', 3],
            ['rerank', 4],
            ['reranker_model', 5]
        ]);

        const orderedEntries = [...entries].sort((left, right) => {
            return (priority.get(left[0]) ?? 100) - (priority.get(right[0]) ?? 100);
        });

        const results = [];

        for (const [key, value] of orderedEntries) {
            try {
                const result = await this.request('/config/set', {
                    method: 'POST',
                    body: JSON.stringify({
                        key: this.mapConfigKey(key),
                        value,
                        persist: true
                    })
                });
                results.push({ ...result, requested_key: key });
            } catch (error) {
                throw new Error(`Failed to save ${key}: ${error.message}`);
            }
        }

        return {
            status: 'success',
            applied: orderedEntries.map(([key]) => key),
            results
        };
    }

    // Governance
    async getAuditLogs(limit = 50) {
        return this.request(`/governance/audit?limit=${limit}`);
    }
}

export default new AxonAPI();
