"""
Migration script for Axon.

This script helps export data from existing RAG systems
and import it into this fully open-source implementation.
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))


def export_from_legacy(output_file: str = "rag_export.json"):
    """
    Example export function from a legacy system.
    """
    print("📤 Exporting from legacy system...")
    
    # Implementation depends on the source system.
    documents = []
    
    kb_path = Path("./knowledge_base")
    if kb_path.exists():
        print(f"   Scanning {kb_path}...")
        for file_path in kb_path.rglob("*.md"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc = {
                "id": str(file_path.relative_to(kb_path)),
                "text": content,
                "metadata": {"source": str(file_path), "type": "markdown"}
            }
            documents.append(doc)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2)
    
    print(f"✅ Exported {len(documents)} documents to {output_file}")
    return documents


def import_to_local_brain(input_file: str = "rag_export.json", directory: str = None):
    """
    Import documents into the Axon.
    """
    from axon.main import OpenStudioBrain
    import asyncio
    
    brain = OpenStudioBrain()
    
    if directory:
        print(f"📥 Importing directory: {directory}...")
        asyncio.run(brain.load_directory(directory))
    else:
        print(f"📥 Importing export file: {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        print(f"   Loaded {len(documents)} documents")
        brain.ingest(documents)
    
    print(f"✅ Import complete.")


def compare_query(query: str):
    """
    Run the same query and display results.
    """
    print(f"\n🔍 Querying: '{query}'")
    print("=" * 60)
    
    try:
        from axon.main import OpenStudioBrain
        brain = OpenStudioBrain()
        response = brain.query(query)
        print(response)
    except Exception as e:
        print(f"   Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate documents to Axon")
    parser.add_argument('action', choices=['export', 'import', 'migrate', 'compare'], help='Action to perform')
    parser.add_argument('--input', default='rag_export.json', help='Input file for import')
    parser.add_argument('--dir', help='Directory to import directly')
    parser.add_argument('--output', default='rag_export.json', help='Output file for export')
    parser.add_argument('--query', default='What are the main topics in my data?', help='Query for comparison')
    
    args = parser.parse_args()
    
    if args.action == 'export':
        export_from_legacy(args.output)
    elif args.action == 'import':
        import_to_local_brain(args.input, args.dir)
    elif args.action == 'migrate':
        documents = export_from_legacy(args.output)
        if documents: import_to_local_brain(args.output)
    elif args.action == 'compare':
        compare_query(args.query)


if __name__ == "__main__":
    main()
