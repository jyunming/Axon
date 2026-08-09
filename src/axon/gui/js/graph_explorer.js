/**
 * Axon Graph Explorer
 * Interactive 3D visualization of the knowledge graph.
 *
 * Features:
 *  - Colours nodes by their semantic entity type (PERSON, ORGANIZATION,
 *    GEO, EVENT, CONCEPT, PRODUCT, UNKNOWN, ...) from GraphRAG extraction.
 *  - Entity search: highlight a matched node, emphasise its neighbourhood,
 *    dim everything else, and show BFS hop-distance to the rest of the graph.
 */

class AxonGraph {
    constructor(api, container) {
        this.api = api;
        this.container = container;
        this.graph = null;
        this.data = { nodes: [], links: [] };
        this._adj = null;   // adjacency map (lazily rebuilt per search)
        this._hl = null;    // active highlight state: { targetId, hops, matches }

        // Colour map keyed by UPPER-CASE entity type. Real graph nodes carry
        // semantic types from GraphRAG extraction, so we key on those; the
        // legacy chunk/entity/community keys are kept for other backends.
        this.colors = {
            PERSON: '#4f9dde',
            ORGANIZATION: '#e59a3c',
            GEO: '#5bbf7b',
            EVENT: '#e5556f',
            CONCEPT: '#3fb0a0',
            PRODUCT: '#d9a441',
            UNKNOWN: '#9aa4ad',
            CHUNK: '#00d2ff',
            ENTITY: '#e91e63',
            COMMUNITY: '#ffc107',
            DEFAULT: '#9aa4ad'
        };

        // Bind style accessors so re-applying them re-triggers a redraw.
        this._nodeColor = this._nodeColor.bind(this);
        this._nodeVal = this._nodeVal.bind(this);
        this._linkColor = this._linkColor.bind(this);
        this._linkWidth = this._linkWidth.bind(this);
    }

    async init() {
        if (!ForceGraph3D) {
            console.error('3D Force Graph library not loaded');
            return;
        }
        this.graph = ForceGraph3D()(this.container)
            .nodeLabel(node => node.tooltip || node.name || node.label || node.id)
            .nodeColor(this._nodeColor)
            .nodeVal(this._nodeVal)
            .nodeOpacity(0.95)
            .linkColor(this._linkColor)
            .linkWidth(this._linkWidth)
            .linkOpacity(0.55)
            .backgroundColor('#0b0b10')
            .showNavInfo(true)
            .onNodeClick(node => {
                const input = document.getElementById('graph-search-input');
                if (input) input.value = node.name || node.label || node.id || '';
                this._highlight(node, new Set([node.id]));
                this._setStatus(`Highlighted "${node.name || node.id}".`, false);
            });
        this.setupSearchUI();
        await this.refresh();
    }

    async refresh() {
        if (!this.graph) return;
        try {
            const data = await this.api.getGraphData();
            if (data && data.nodes) {
                this.data = data;
                this._adj = null;
                this.graph.graphData(this.data);
                this.clearHighlight();
            }
        } catch (error) {
            console.error('Failed to load graph data:', error);
        }
    }

    resize() {
        if (this.graph) {
            this.graph.width(this.container.clientWidth);
            this.graph.height(this.container.clientHeight);
        }
    }

    // ------------------------------------------------------------------ colours

    colorFor(type) {
        const key = String(type || '').toUpperCase();
        return this.colors[key] || this.colors.DEFAULT;
    }

    _hexToRgb(hex) {
        const m = /^#?([0-9a-f]{6})$/i.exec(String(hex));
        if (!m) return null;
        const n = parseInt(m[1], 16);
        return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    }

    /** Blend a hex colour toward the background; factor 1 = full colour, 0 = bg. */
    _mix(hex, factor) {
        const bg = [11, 11, 16];
        const c = this._hexToRgb(hex);
        if (!c) return hex;
        const f = Math.max(0, Math.min(1, factor));
        const r = Math.round(c[0] * f + bg[0] * (1 - f));
        const g = Math.round(c[1] * f + bg[1] * (1 - f));
        const b = Math.round(c[2] * f + bg[2] * (1 - f));
        return `rgb(${r},${g},${b})`;
    }

    // -------------------------------------------------------------- style accessors

    _nodeColor(node) {
        const base = this.colorFor(node.type);
        if (!this._hl) return base;
        if (this._hl.matches.has(node.id)) {
            return node.id === this._hl.targetId ? '#ffd54a' : '#ffcf6b';
        }
        const hop = this._hl.hops.get(node.id);
        if (hop === undefined) return '#171a20'; // unreachable: recede into the background
        // Closer neighbours stay bright; brightness fades with hop distance.
        const factor = Math.max(0.35, 1 - (hop - 1) * 0.22);
        return this._mix(base, factor);
    }

    _nodeVal(node) {
        const base = (Number.isFinite(node.val) && node.val > 0) ? node.val : 4;
        if (!this._hl) return base;
        if (this._hl.matches.has(node.id)) return base * 3 + 8;
        const hop = this._hl.hops.get(node.id);
        if (hop === 1) return base * 1.6;
        if (hop === undefined) return base * 0.5;
        return base;
    }

    _linkColor(link) {
        if (!this._hl) return 'rgba(150,170,205,0.45)';
        const s = this._endId(link.source);
        const t = this._endId(link.target);
        if (s === this._hl.targetId || t === this._hl.targetId) return '#ffd54a';
        const hs = this._hl.hops.get(s);
        const ht = this._hl.hops.get(t);
        if (hs !== undefined && ht !== undefined) return '#5a6b86';
        return '#171a20';
    }

    _linkWidth(link) {
        const base = (Number.isFinite(link.width) && link.width > 0) ? link.width : 0.6;
        if (!this._hl) return base;
        const s = this._endId(link.source);
        const t = this._endId(link.target);
        if (s === this._hl.targetId || t === this._hl.targetId) return Math.max(2.2, base * 2);
        return base;
    }

    _applyStyles() {
        if (!this.graph) return;
        this.graph
            .nodeColor(this._nodeColor)
            .nodeVal(this._nodeVal)
            .linkColor(this._linkColor)
            .linkWidth(this._linkWidth);
    }

    // ------------------------------------------------------------ graph traversal

    /** 3d-force-graph rewrites link.source/target from ids to node objects. */
    _endId(endpoint) {
        return (endpoint && typeof endpoint === 'object') ? endpoint.id : endpoint;
    }

    _buildAdjacency() {
        const adj = new Map();
        const add = (a, b) => {
            if (!adj.has(a)) adj.set(a, new Set());
            adj.get(a).add(b);
        };
        for (const link of (this.data.links || [])) {
            const s = this._endId(link.source);
            const t = this._endId(link.target);
            if (s == null || t == null) continue;
            add(s, t);
            add(t, s);
        }
        return adj;
    }

    /** Breadth-first hop distance from startId to every reachable node. */
    _bfs(startId) {
        const adj = this._adj || (this._adj = this._buildAdjacency());
        const hops = new Map([[startId, 0]]);
        let frontier = [startId];
        let depth = 0;
        while (frontier.length) {
            depth++;
            const next = [];
            for (const id of frontier) {
                for (const nb of (adj.get(id) || [])) {
                    if (!hops.has(nb)) {
                        hops.set(nb, depth);
                        next.push(nb);
                    }
                }
            }
            frontier = next;
        }
        return hops;
    }

    // -------------------------------------------------------------- search + focus

    setupSearchUI() {
        const input = document.getElementById('graph-search-input');
        const clearBtn = document.getElementById('graph-search-clear');
        if (input && !input._axonBound) {
            input._axonBound = true;
            let debounce = null;
            input.addEventListener('input', () => {
                clearTimeout(debounce);
                const value = input.value;
                debounce = setTimeout(() => {
                    if (value.trim()) this.search(value);
                    else this.clearHighlight();
                }, 250);
            });
            input.addEventListener('keydown', (event) => {
                if (event.key === 'Enter') {
                    clearTimeout(debounce);
                    this.search(input.value);
                } else if (event.key === 'Escape') {
                    this.clearHighlight();
                }
            });
        }
        if (clearBtn && !clearBtn._axonBound) {
            clearBtn._axonBound = true;
            clearBtn.addEventListener('click', () => this.clearHighlight());
        }
    }

    search(rawQuery) {
        const query = String(rawQuery || '').trim().toLowerCase();
        if (!query) {
            this.clearHighlight();
            return;
        }
        const nodes = this.data.nodes || [];
        const matches = nodes.filter(node => {
            const name = String(node.name || node.label || node.id || '').toLowerCase();
            return name.includes(query);
        });
        if (matches.length === 0) {
            this._setStatus(`No entity matches "${rawQuery.trim()}".`, true);
            this.clearHighlight(false);
            return;
        }
        const score = (node) => {
            const name = String(node.name || node.label || node.id || '').toLowerCase();
            if (name === query) return 0;
            if (name.startsWith(query)) return 1;
            return 2;
        };
        matches.sort((a, b) =>
            score(a) - score(b)
            || String(a.name || a.id).length - String(b.name || b.id).length
            || (b.degree || 0) - (a.degree || 0));
        const target = matches[0];
        this._highlight(target, new Set(matches.map(m => m.id)));
        const extra = matches.length > 1
            ? ` (+${matches.length - 1} more match${matches.length - 1 > 1 ? 'es' : ''})`
            : '';
        this._setStatus(`Highlighted "${target.name || target.id}"${extra}.`, false);
    }

    _highlight(target, matchIds) {
        this._adj = this._buildAdjacency();
        const hops = this._bfs(target.id);
        this._hl = { targetId: target.id, hops, matches: matchIds || new Set([target.id]) };
        this._applyStyles();
        this._focus(target);
        this._renderDistances(target, hops);
    }

    clearHighlight(clearInput = true) {
        this._hl = null;
        this._applyStyles();
        const panel = document.getElementById('graph-distance-panel');
        if (panel) panel.classList.add('hidden');
        if (clearInput) {
            const input = document.getElementById('graph-search-input');
            if (input) input.value = '';
            this._setStatus('', false);
        }
        if (this.graph && typeof this.graph.zoomToFit === 'function') {
            try { this.graph.zoomToFit(600, 40); } catch (_e) { /* ignore */ }
        }
    }

    _focus(node) {
        if (!this.graph) return;
        const distance = 90;
        const x = Number.isFinite(node.x) ? node.x : 0;
        const y = Number.isFinite(node.y) ? node.y : 0;
        const z = Number.isFinite(node.z) ? node.z : 0;
        const radius = Math.hypot(x, y, z);
        const safeRadius = radius > 0 ? radius : 1;
        const distRatio = 1 + distance / safeRadius;
        this.graph.cameraPosition(
            { x: x * distRatio, y: y * distRatio, z: z * distRatio },
            { x, y, z },
            1400
        );
    }

    _setStatus(message, isError) {
        const el = document.getElementById('graph-search-status');
        if (!el) return;
        el.textContent = message || '';
        el.classList.toggle('error', Boolean(isError && message));
    }

    _renderDistances(target, hops) {
        const panel = document.getElementById('graph-distance-panel');
        const list = document.getElementById('graph-distance-list');
        const targetLabel = document.getElementById('graph-distance-target');
        if (!panel || !list) return;
        if (targetLabel) targetLabel.textContent = target.name || target.id || '';

        const byId = new Map((this.data.nodes || []).map(n => [n.id, n]));
        const rows = [];
        hops.forEach((hop, id) => {
            if (hop === 0) return; // skip the target itself
            const node = byId.get(id);
            if (node) rows.push({ node, hop });
        });
        rows.sort((a, b) =>
            a.hop - b.hop
            || String(a.node.name || a.node.id).localeCompare(String(b.node.name || b.node.id)));

        list.replaceChildren();
        if (rows.length === 0) {
            const empty = document.createElement('li');
            empty.className = 'graph-distance-empty';
            empty.textContent = 'Isolated node — no connected neighbours.';
            list.appendChild(empty);
        } else {
            rows.slice(0, 20).forEach(({ node, hop }) => {
                const item = document.createElement('li');
                item.className = 'graph-distance-item';
                item.title = `Focus ${node.name || node.id}`;

                const badge = document.createElement('span');
                badge.className = `hop-badge hop-${Math.min(hop, 4)}`;
                badge.textContent = `${hop}`;
                badge.title = `${hop} hop${hop > 1 ? 's' : ''} away`;

                const name = document.createElement('span');
                name.className = 'hop-name';
                name.textContent = node.name || node.label || node.id;

                const type = document.createElement('span');
                type.className = 'hop-type';
                type.textContent = node.type || '';

                item.append(badge, name, type);
                item.addEventListener('click', () => this._focus(node));
                list.appendChild(item);
            });
            if (rows.length > 20) {
                const more = document.createElement('li');
                more.className = 'graph-distance-empty';
                more.textContent = `+ ${rows.length - 20} more reachable entities`;
                list.appendChild(more);
            }
        }
        panel.classList.remove('hidden');
    }
}

export default AxonGraph;
