"""Maintenance state, Copilot bridge, and LLM task routes."""
from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from axon.api_schemas import (
    _VALID_PROJECT_NAME_RE,
    CopilotAgentRequest,
    CopilotTaskResult,
    MaintenanceStateRequest,
)

logger = logging.getLogger("AxonAPI")
router = APIRouter()


@router.post("/copilot/agent")
async def copilot_agent_handler(
    request: Request,
    body: CopilotAgentRequest,
):
    """Handle chat requests from GitHub Copilot."""
    from axon import api as _api
    from axon.projects import list_projects as _list_projects

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    user_query = body.messages[-1].content if body.messages else ""
    if not user_query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    parts = user_query.strip().split(maxsplit=1)
    command = parts[0].lower() if parts[0].startswith("/") else None
    args = parts[1] if len(parts) > 1 else ""

    async def response_stream():
        yield f"data: {json.dumps({'type': 'created', 'id': body.agent_request_id})}\n\n"

        try:
            if command == "/search":
                yield f"data: {json.dumps({'type': 'text', 'content': f'🔍 Searching Axon for: {args}...'})}\n\n"
                retrieval_data = await asyncio.to_thread(brain._execute_retrieval, args)
                results = retrieval_data["results"]
                if not results:
                    content = "No relevant documents found."
                else:
                    content = "### Search Results\n\n"
                    for i, r in enumerate(results[:5]):
                        content += f"**{i+1}. {os.path.basename((r.get('metadata') or {}).get('source', 'unknown'))}** (Score: {r['score']:.2f})\n"
                        content += f"> {r['text'][:200]}...\n\n"
                yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"

            elif command == "/ingest":
                yield f"data: {json.dumps({'type': 'text', 'content': f'📥 Ingesting URL: {args}...'})}\n\n"
                from axon.loaders import URLLoader

                loader = URLLoader()
                docs = await asyncio.to_thread(loader.load, args)
                if docs:
                    await asyncio.to_thread(brain.ingest, docs)
                    yield f"data: {json.dumps({'type': 'text', 'content': f'✅ Successfully ingested: {args}'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'text', 'content': f'❌ Failed to ingest: {args}'})}\n\n"

            elif command == "/projects":
                projects = _list_projects()
                content = "### Available Axon Projects\n\n"
                for p in projects:
                    content += f"- **{p['name']}**: {p.get('description', 'No description')}\n"
                yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"

            else:
                answer = await asyncio.to_thread(
                    brain.query,
                    user_query,
                    chat_history=[
                        {"role": m.role, "content": m.content} for m in body.messages[:-1]
                    ],
                )
                if isinstance(answer, dict):
                    answer = answer.get("response", answer.get("answer", str(answer)))
                yield f"data: {json.dumps({'type': 'text', 'content': answer})}\n\n"

        except Exception as e:
            logger.error(f"Copilot Agent error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(response_stream(), media_type="text/event-stream")


@router.get("/llm/copilot/tasks")
async def get_copilot_tasks():
    """Poll for pending LLM tasks intended for the VS Code Copilot bridge."""
    from axon.main import _copilot_bridge_lock, _copilot_task_queue

    with _copilot_bridge_lock:
        tasks = list(_copilot_task_queue)
        _copilot_task_queue.clear()
    return {"tasks": tasks}


@router.post("/llm/copilot/result/{task_id}")
async def submit_copilot_result(task_id: str, body: CopilotTaskResult):
    """Submit the result of a Copilot LLM task back to the backend."""
    from axon.main import _copilot_bridge_lock, _copilot_responses

    with _copilot_bridge_lock:
        if task_id in _copilot_responses:
            _copilot_responses[task_id]["result"] = body.result
            _copilot_responses[task_id]["error"] = body.error
            _copilot_responses[task_id]["event"].set()
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="Task not found or expired")


@router.post("/project/maintenance")
async def set_project_maintenance(request: MaintenanceStateRequest, req: Request):
    """Set the maintenance state of a project."""
    from axon import governance as gov
    from axon.maintenance import apply_maintenance_state

    if not _VALID_PROJECT_NAME_RE.match(request.name):
        raise HTTPException(status_code=422, detail=f"Invalid project name: '{request.name}'")
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    try:
        result = apply_maintenance_state(request.name, request.state)
        gov.emit(
            "maintenance_changed",
            "project",
            request.name,
            project=request.name,
            details={"state": request.state, "result": result},
            surface=surface,
            request_id=rid,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404 if "does not exist" in str(e) else 422, detail=str(e))
    except Exception as e:
        logger.error(f"set_maintenance_state failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/project/maintenance")
async def get_project_maintenance(name: str):
    """Get the current maintenance state of a project."""
    from axon.maintenance import get_maintenance_status
    from axon.projects import project_dir

    if not _VALID_PROJECT_NAME_RE.match(name):
        raise HTTPException(status_code=422, detail=f"Invalid project name: '{name}'")
    if not (project_dir(name) / "meta.json").exists():
        raise HTTPException(status_code=404, detail=f"Project '{name}' not found.")
    return get_maintenance_status(name)
