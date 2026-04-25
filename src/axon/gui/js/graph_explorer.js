/**
 * Axon Graph Explorer
 * Interactive 3D visualization of the knowledge graph.
 */

class AxonGraph {
    constructor(api, container) {
        this.api = api;
        this.container = container;
        this.graph = null;
        this.data = { nodes: [], links: [] };
        this.colors = {
            chunk: '#00d2ff',
            entity: '#e91e63',
            community: '#ffc107',
            default: '#999'
        };
    }
    async init() {
        if (!ForceGraph3D) {
            console.error('3D Force Graph library not loaded');
            return;
        }
        this.graph = ForceGraph3D()(this.container)
            .nodeLabel('label')
            .nodeAutoColorBy('type')
            .nodeColor(node => this.colors[node.type] || this.colors.default)
            .linkOpacity(0.3)
            .linkWidth(0.5)
            .backgroundColor('#0b0b10')
            .showNavInfo(true)
            .onNodeClick(node => {
                // Focus camera on node
                const distance = 40;
                const x = Number.isFinite(node.x) ? node.x : 0;
                const y = Number.isFinite(node.y) ? node.y : 0;
                const z = Number.isFinite(node.z) ? node.z : 0;
                const radius = Math.hypot(x, y, z);
                const safeRadius = radius > 0 ? radius : 1;
                const distRatio = 1 + distance / safeRadius;
                this.graph.cameraPosition(
                    { x: x * distRatio, y: y * distRatio, z: z * distRatio },
                    { x, y, z },
                    1500
                );
            });
        await this.refresh();
    }
    async refresh() {
        if (!this.graph) return;
        try {
            const data = await this.api.getGraphData();
            if (data && data.nodes) {
                this.data = data;
                this.graph.graphData(this.data);
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
}

export default AxonGraph;
