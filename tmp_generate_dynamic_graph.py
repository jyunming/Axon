#!/usr/bin/env python3
"""Generate a small dynamic graph payload using DynamicGraphBackend and write
JSON and HTML artifacts into the session artifacts folder.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from axon.graph_backends.dynamic_graph_backend import DynamicGraphBackend


class FakeLLM:
    def complete(self, prompt, **kwargs):
        # Return deterministic extraction outputs for entity and fact prompts
        if "Extract the key named entities" in prompt:
            return (
                "Alice | PERSON | CEO of ExampleCorp\n"
                "Bob | PERSON | CTO of ExampleCorp\n"
                "ExampleCorp | ORGANIZATION | ExampleCorp is a company"
            )
        if "Extract key relationships" in prompt:
            return (
                "Alice | WORKS_FOR | ExampleCorp | works at ExampleCorp | 9\n"
                "Bob | WORKS_FOR | ExampleCorp | works at ExampleCorp | 8"
            )
        return ""


def main():
    # Prepare a simple fake brain with config and llm
    tmp_base = Path("tmp_dynamic_graph")
    tmp_base.mkdir(exist_ok=True)

    brain = SimpleNamespace()
    brain.llm = FakeLLM()
    brain.config = SimpleNamespace(bm25_path=str(tmp_base))

    backend = DynamicGraphBackend(brain)

    # Ingest a couple of simple chunks
    chunks = [
        {"id": "c1", "text": "Alice and Bob work at ExampleCorp."},
        {"id": "c2", "text": "ExampleCorp is based in Metropolis."},
    ]

    backend.ingest(chunks)

    payload = backend.graph_data()

    # Write artifacts to the session-state artifacts folder
    artifacts_dir = Path(os.path.expanduser("~")) / ".copilot" / "session-state" / "b7762e66-34dc-4b83-b547-89193ce7e228" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    json_path = artifacts_dir / "dynamic_graph.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"nodes": payload.nodes, "links": payload.links}, f, indent=2)

    # Simple HTML preview embedding the JSON inline (no external deps)
    html_path = artifacts_dir / "dynamic_graph.html"
    html_template = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Dynamic Graph</title>"
        "<style>svg{width:100%;height:80vh;border:1px solid #ccc}</style></head><body>"
        "<h3>Dynamic Graph</h3><div id=\"count\"></div><svg id=\"svg\"></svg>"
        "<script>const data = "
        + json.dumps({"nodes": payload.nodes, "links": payload.links})
        + ";document.getElementById('count').innerText = `nodes: ${data.nodes.length}, links: ${data.links.length}`;"
        "const svg=document.getElementById('svg');const width=800;const height=600;svg.setAttribute('viewBox',`0 0 ${width} ${height}`);const NS='http://www.w3.org/2000/svg';const g=document.createElementNS(NS,'g');svg.appendChild(g);const nodes=data.nodes;const links=data.links;const nodeMap={};nodes.forEach((n,i)=>{nodeMap[n.id]=Object.assign({x:60+Math.random()*(width-120),y:60+Math.random()*(height-120)},n)});links.forEach(l=>{const line=document.createElementNS(NS,'line');line.setAttribute('x1',nodeMap[l.source].x);line.setAttribute('y1',nodeMap[l.source].y);line.setAttribute('x2',nodeMap[l.target].x);line.setAttribute('y2',nodeMap[l.target].y);line.setAttribute('stroke','#999');line.setAttribute('stroke-width',Math.max(1,Math.floor(l.weight*4)));g.appendChild(line);});nodes.forEach(n=>{const c=document.createElementNS(NS,'circle');c.setAttribute('cx',nodeMap[n.id].x);c.setAttribute('cy',nodeMap[n.id].y);c.setAttribute('r',8);c.setAttribute('fill','#2b7');g.appendChild(c);const t=document.createElementNS(NS,'text');t.setAttribute('x',nodeMap[n.id].x+10);t.setAttribute('y',nodeMap[n.id].y+4);t.textContent=n.name;g.appendChild(t);});</script></body></html>"
    )
    html_path.write_text(html_template, encoding="utf-8")

    print("Wrote:", json_path, html_path)


if __name__ == '__main__':
    main()
