"""
Simple Agent Orchestration Example.
This script demonstrates how an agent can use the Axon API.
"""

import httpx
import json

API_URL = "http://localhost:8000"

def call_brain_query(query: str):
    """Simple wrapper to call the brain's query endpoint."""
    print(f"🤖 Agent: Querying brain for '{query}'...")
    with httpx.Client() as client:
        response = client.post(f"{API_URL}/query", json={"query": query}, timeout=60.0)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.text}"

def call_brain_search(query: str, top_k: int = 3):
    """Simple wrapper to call the brain's raw search endpoint."""
    print(f"🔍 Agent: Searching raw documents for '{query}'...")
    with httpx.Client() as client:
        response = client.post(f"{API_URL}/search", json={"query": query, "top_k": top_k}, timeout=60.0)
        if response.status_code == 200:
            return response.json()
        else:
            return []

def main():
    print("🚀 Starting Simple Agent Loop")
    print("-" * 40)
    
    # Example 1: Direct Query
    answer = call_brain_query("What is the core purpose of this system?")
    print(f"📝 Answer: {answer}")
    print("-" * 40)
    
    # Example 2: Multi-step (Search then Reasoning)
    print("🤔 Agent: I need to check the raw documents first...")
    docs = call_brain_search("multimodal support")
    
    if docs:
        print(f"📄 Found {len(docs)} relevant documents.")
        # An agent would normally pass these docs back to an LLM for reasoning
        # Here we just show the output.
        for i, doc in enumerate(docs):
            print(f"   [{i+1}] (Score: {doc['score']:.2f}) {doc['text'][:100]}...")
    else:
        print("❌ No documents found.")

if __name__ == "__main__":
    main()
