"""Ingest, delete, collection, and refresh routes."""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from axon.api_routes import enforce_project as _enforce_project
from axon.api_schemas import (
    BatchTextIngestRequest,
    DeleteRequest,
    IngestRequest,
    TextIngestRequest,
    URLIngestRequest,
    _validate_ingest_path,
)

logger = logging.getLogger("AxonAPI")
router = APIRouter()


@router.post("/ingest/refresh")
async def refresh_docs():
    """Re-ingest any tracked files whose content has changed since last ingest."""
    import functools
    import hashlib as _hashlib

    from axon import api as _api
    from axon.loaders import DirectoryLoader

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    versions = brain.get_doc_versions()
    results: dict[str, list] = {"skipped": [], "reingested": [], "missing": [], "errors": []}
    for source_id, record in versions.items():
        if not os.path.exists(source_id):
            results["missing"].append(source_id)
            continue
        try:
            loop = asyncio.get_running_loop()
            loader = DirectoryLoader()
            suffix = os.path.splitext(source_id)[1].lower()
            loader_instance = loader.loaders.get(suffix)
            if loader_instance is None:
                results["errors"].append(
                    {"source": source_id, "error": f"no loader for extension '{suffix}'"}
                )
                continue
            docs = await loop.run_in_executor(
                None, functools.partial(loader_instance.load, source_id)
            )
            if not docs:
                results["errors"].append(
                    {"source": source_id, "error": "loader returned no documents"}
                )
                continue
            combined = "".join(d.get("text", "") for d in docs)
            current_hash = _hashlib.md5(combined.encode("utf-8", errors="replace")).hexdigest()
            if current_hash == record.get("content_hash"):
                results["skipped"].append(source_id)
                continue
            await loop.run_in_executor(None, functools.partial(brain.ingest, docs))
            results["reingested"].append(source_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Exception as exc:
            results["errors"].append({"source": source_id, "error": str(exc)})
    return results


@router.post("/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks, req: Request):
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    validated_path = _validate_ingest_path(request.path)

    try:
        requested_path = pathlib.Path(os.path.realpath(validated_path))
        if not requested_path.exists():
            raise HTTPException(status_code=404, detail="Path does not exist")
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")

    try:
        brain._assert_write_allowed("ingest")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    _api._evict_old_jobs()
    _api._jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "phase": "loading",
        "path": str(requested_path),
        "started_at": now.isoformat(),
        "started_at_ts": now.timestamp(),
        "completed_at": None,
        "files_total": None,
        "chunks_total": None,
        "chunks_embedded": None,
        "documents_ingested": None,
        "error": None,
    }

    rid = getattr(req.state, "request_id", job_id)
    surface = getattr(req.state, "surface", "api")

    def _make_progress_callback(job: dict):
        def _cb(phase: str, **kwargs) -> None:
            job["phase"] = phase
            if "files_total" in kwargs:
                job["files_total"] = kwargs["files_total"]
            if "chunks_total" in kwargs:
                job["chunks_total"] = kwargs["chunks_total"]
            if "chunks_embedded" in kwargs:
                job["chunks_embedded"] = kwargs["chunks_embedded"]

        return _cb

    def process_ingestion():
        from axon import governance as gov
        from axon.loaders import DirectoryLoader

        project = getattr(brain, "_active_project", "default")
        gov.emit(
            "ingest_started",
            "file",
            str(requested_path),
            project=project,
            status="started",
            details={"job_id": job_id},
            surface=surface,
            request_id=rid,
        )
        try:
            _api._jobs[job_id]["phase"] = "loading"
            loader_mgr = DirectoryLoader()
            if requested_path.is_dir():
                docs = loader_mgr.load(str(requested_path))
            else:
                ext = requested_path.suffix.lower()
                if ext in loader_mgr.loaders:
                    docs = loader_mgr.loaders[ext].load(str(requested_path))
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    docs = []
            _api._jobs[job_id]["files_total"] = len(docs)
            if docs:
                brain.ingest(docs, progress_callback=_make_progress_callback(_api._jobs[job_id]))
            _api._jobs[job_id]["status"] = "completed"
            _api._jobs[job_id]["phase"] = "completed"
            _api._jobs[job_id]["documents_ingested"] = len(docs)
            _api._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            gov.emit(
                "ingest_completed",
                "file",
                str(requested_path),
                project=project,
                details={"job_id": job_id, "documents_ingested": len(docs)},
                surface=surface,
                request_id=rid,
            )
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            _api._jobs[job_id]["status"] = "failed"
            _api._jobs[job_id]["phase"] = "failed"
            _api._jobs[job_id]["error"] = str(e)
            _api._jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
            gov.emit(
                "ingest_failed",
                "file",
                str(requested_path),
                project=project,
                status="failed",
                details={"job_id": job_id, "error": "ingest error"},
                surface=surface,
                request_id=rid,
            )

    background_tasks.add_task(process_ingestion)
    return {
        "message": f"Ingestion started for {validated_path}",
        "status": "processing",
        "job_id": job_id,
    }


@router.get("/ingest/status/{job_id}")
async def get_ingest_status(job_id: str):
    """Poll the status of an async ingest job started by POST /ingest."""
    from axon import api as _api

    brain = _api.brain
    job = _api._jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have already been evicted (TTL: 60 min) or the job_id is invalid.",
        )
    result = {k: v for k, v in job.items() if k != "started_at_ts"}
    result["community_build_in_progress"] = bool(
        brain and getattr(brain, "_community_build_in_progress", False)
    )
    return result


@router.get("/tracked-docs")
async def list_tracked_docs():
    """List all tracked document sources with metadata."""
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    return {"docs": brain.get_doc_versions()}


@router.get("/collection")
async def get_collection():
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        docs = brain.list_documents()
        return {
            "total_files": len(docs),
            "total_chunks": sum(d["chunks"] for d in docs),
            "files": [{"source": d["source"], "chunks": d["chunks"]} for d in docs],
        }
    except Exception as e:
        logger.error(f"Error listing collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/stale")
async def get_stale_docs(days: int = 7):
    """Return documents that have not been re-ingested within *days* calendar days."""
    from axon import api as _api

    if days < 0:
        raise HTTPException(status_code=400, detail="'days' must be >= 0.")

    cutoff = datetime.now(timezone.utc).timestamp() - days * 86_400
    stale: list[dict] = []
    for project_name, hashes in _api._source_hashes.items():
        for _content_hash, meta in hashes.items():
            try:
                ingested_ts = datetime.fromisoformat(meta["last_ingested_at"]).timestamp()
            except (KeyError, ValueError):
                continue
            if ingested_ts < cutoff:
                stale.append(
                    {
                        "doc_id": meta["doc_id"],
                        "project": project_name,
                        "last_ingested_at": meta["last_ingested_at"],
                        "age_days": round(
                            (datetime.now(timezone.utc).timestamp() - ingested_ts) / 86_400, 1
                        ),
                    }
                )
    return {"stale_docs": stale, "total": len(stale), "threshold_days": days}


@router.post("/add_text")
async def add_text(request: TextIngestRequest):
    from axon import api as _api

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(request.project, brain)

    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    doc_id = request.doc_id or f"agent_doc_{uuid.uuid4().hex[:8]}"
    project_key = request.project or brain._active_project

    skip = _api._check_dedup(request.text, project_key)
    if skip:
        return {**skip, "doc_id": skip["doc_id"]}

    from axon.loaders import SmartTextLoader

    loader = SmartTextLoader()
    documents = loader.load_text(request.text, source=doc_id)

    if request.metadata:
        for doc in documents:
            doc["metadata"].update(request.metadata)

    try:
        brain.ingest(documents)
        _api._record_dedup(request.text, doc_id, project_key)
        return {"status": "success", "doc_id": doc_id, "chunks": len(documents)}
    except Exception as e:
        logger.error(f"Error adding text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add_texts")
async def add_texts(request: BatchTextIngestRequest):
    """Ingest a list of documents in a single embedding batch."""
    from axon import api as _api
    from axon.api_schemas import _compute_content_hash
    from axon.loaders import SmartTextLoader

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(request.project, brain)

    results: list[dict] = []
    docs_to_ingest: list[dict] = []
    project_key = request.project or brain._active_project
    pending_records: list[tuple[str, str]] = []
    batch_hashes: dict[str, str] = {}
    loader = SmartTextLoader()

    for item in request.docs:
        doc_id = item.doc_id or f"agent_doc_{uuid.uuid4().hex[:8]}"
        skip = _api._check_dedup(item.text, project_key)
        if skip:
            results.append({"id": skip["doc_id"], "status": "skipped", "error": None})
            continue
        content_hash = _compute_content_hash(item.text)
        if content_hash in batch_hashes:
            results.append({"id": batch_hashes[content_hash], "status": "skipped", "error": None})
            continue
        batch_hashes[content_hash] = doc_id

        item_docs = loader.load_text(item.text, source=doc_id)
        if item.metadata:
            for d in item_docs:
                d["metadata"].update(item.metadata)

        docs_to_ingest.extend(item_docs)
        pending_records.append((doc_id, item.text))
        results.append({"id": doc_id, "status": "created", "chunks": len(item_docs)})

    if docs_to_ingest:
        try:
            brain.ingest(docs_to_ingest)
            for doc_id, text in pending_records:
                _api._record_dedup(text, doc_id, project_key)
        except Exception as e:
            logger.error(f"Error batch-ingesting texts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return results


@router.post("/ingest_url")
async def ingest_url(request: URLIngestRequest):
    """Fetch an HTTP/HTTPS URL and ingest its text content."""
    from axon import api as _api
    from axon.loaders import URLLoader

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(request.project, brain)

    loader = URLLoader()
    project_key = request.project or brain._active_project
    try:
        docs = loader.load(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected error fetching URL '{request.url}': {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if not docs:
        raise HTTPException(status_code=422, detail="No content could be extracted from the URL.")

    doc = docs[0]
    if request.metadata:
        doc["metadata"].update(request.metadata)

    skip = _api._check_dedup(doc["text"], project_key)
    if skip:
        return {**skip, "doc_id": skip["doc_id"], "url": request.url}

    try:
        brain.ingest([doc])
        _api._record_dedup(doc["text"], doc["id"], project_key)
    except Exception as exc:
        logger.error(f"Error ingesting URL content: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return {"status": "ingested", "doc_id": doc["id"], "url": request.url}


@router.post("/delete")
async def delete_documents(request: DeleteRequest, req: Request):
    from axon import api as _api
    from axon import governance as gov

    brain = _api.brain
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    project = getattr(brain, "_active_project", "default")
    try:
        brain._assert_write_allowed("delete")
        existing = brain.vector_store.get_by_ids(request.doc_ids)
        existing_ids_set = {doc["id"] for doc in existing}
        existing_ids = [i for i in request.doc_ids if i in existing_ids_set]
        not_found = [i for i in request.doc_ids if i not in existing_ids_set]
        if existing_ids:
            brain.vector_store.delete_by_ids(existing_ids)
            if brain.bm25 is not None:
                brain.bm25.delete_documents(existing_ids)
            if brain._entity_graph:
                brain._prune_entity_graph(set(existing_ids))
        gov.emit(
            "delete",
            "document",
            ",".join(existing_ids[:10]),
            project=project,
            details={"deleted": len(existing_ids), "not_found": len(not_found)},
            surface=surface,
            request_id=rid,
        )
        return {
            "status": "success",
            "deleted": len(existing_ids),
            "doc_ids": existing_ids,
            "not_found": not_found,
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
