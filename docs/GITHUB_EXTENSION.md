# GitHub Copilot Extension: Axon Integration

This guide explains how to set up Axon as a server-side GitHub Copilot Extension. This allows you to use `@axon` in any GitHub Copilot interface (Web, IDE, Mobile) without installing a local VS Code extension.

## 1. Prerequisites
- Axon API running on a public URL (or accessible via a tunnel like `ngrok`).
- A GitHub account with access to Copilot.

## 2. Create a GitHub App
1. Go to your **GitHub Settings** -> **Developer settings** -> **GitHub Apps** -> **New GitHub App**.
2. **Name**: `Axon-RAG` (or your preferred name).
3. **Homepage URL**: Your Axon instance URL.
4. **Callback URL**: `https://<your-axon-url>/copilot/callback`
5. **Permissions**:
   - `Copilot Chat`: Read-only (under "Account permissions").
6. **Copilot Agent Settings**:
   - **Agent URL**: `https://<your-axon-url>/copilot/agent`
   - **Pre-defined commands**: `/search`, `/ingest`, `/projects`.

## 3. Configuration
Set the following environment variables on your Axon server:
```bash
GITHUB_APP_ID=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
```

## 4. Usage in Copilot Chat
Once the app is installed, you can simply type:
- `@axon What is the current project status?`
- `@axon /search "Quantum Mechanics"`
- `@axon /ingest https://example.com/docs`
- `@axon /projects`

## 5. Why use this over the VSIX?
- **Zero Install**: Works everywhere Copilot works (including github.com).
- **Organization Wide**: Admins can install it for the whole team instantly.
- **SOTA Support**: Brings Axon's **GraphRAG**, **RAPTOR**, and **Table-Aware** retrieval into the native Copilot experience.
