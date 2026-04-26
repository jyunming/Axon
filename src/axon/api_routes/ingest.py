"""Ingest, delete, collection, and refresh routes."""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import tempfile
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile

from axon.api_routes import _enforce_write_access
from axon.api_routes import enforce_project as _enforce_project
from axon.api_routes._rate_limit import enforce_rate_limit
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


# Module-level fallback caps used before the brain config is available.
# Real values come from AxonConfig.max_upload_bytes / .max_files_per_request
# which are proper dataclass fields (not getattr defaults).
MAX_UPLOAD_BYTES: int = 500 * 1024 * 1024  # 500 MiB per file
MAX_FILES_PER_REQUEST: int = 1000


def _project_label(brain: object, request_project: str | None = None) -> str:
    """Return a stable project label for Prometheus counters."""
    return request_project or getattr(brain, "_active_project", None) or "_global"


_PATH_ENRICHMENT_EXCLUDED_TYPES = frozenset(
    {
        "csv",
        "tsv",
        "excel",
        "parquet",
        "image",
        "code",
    }
)


def _normalise_uploaded_filename(filename: str | None, index: int) -> str:
    """Return a safe basename for an uploaded file, preserving its extension."""
    raw_name = pathlib.Path(filename or "").name.strip()
    if not raw_name:
        raw_name = f"upload_{index}"
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw_name)
    safe_stem = safe_stem.strip("._") or f"upload_{index}"
    return safe_stem


@router.post("/ingest/refresh")
async def refresh_docs(background_tasks: BackgroundTasks):
    """Re-ingest tracked files whose content has changed.
    Returns a job_id immediately; the refresh runs in the background.
    Poll ``GET /ingest/status/{job_id}`` until ``status == "completed"``.
    """
    import functools
    import hashlib as _hashlib

    from axon import api as _api
    from axon.loaders import DirectoryLoader

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_write_access(brain, "refresh")
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc)
    _api._evict_old_jobs()
    _api._jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "phase": "scanning",
        "started_at": now.isoformat(),
        "started_at_ts": now.timestamp(),
        "completed_at": None,
        "reingested": [],
        "skipped": [],
        "missing": [],
        "errors": [],
        "error": None,
    }

    def _run_refresh() -> None:
        job = _api._jobs.get(job_id)
        if job is None:
            return
        try:
            versions = brain.get_doc_versions() or {}
            loader = DirectoryLoader()
            for source_id, record in versions.items():
                try:
                    src_path = pathlib.Path(source_id).expanduser().resolve()
                except Exception:
                    job["errors"].append({"source": source_id, "error": "invalid source path"})
                    continue
                if not src_path.exists():
                    job["missing"].append(source_id)
                    continue
                try:
                    _validate_ingest_path(str(src_path))
                except (ValueError, PermissionError) as e:
                    logger.warning(
                        "Refresh: skipping %s (path validation failed: %s)", source_id, e
                    )
                    continue
                try:
                    suffix = os.path.splitext(str(src_path))[1].lower()
                    loader_instance = loader.loaders.get(suffix)
                    if loader_instance is None:
                        job["errors"].append(
                            {"source": source_id, "error": f"no loader for extension '{suffix}'"}
                        )
                        continue
                    docs = functools.partial(loader_instance.load, str(src_path))()
                    if not docs:
                        job["errors"].append(
                            {"source": source_id, "error": "loader returned no documents"}
                        )
                        continue
                    combined = "".join(d.get("text", "") for d in docs)
                    # NOTE: MD5 retained here for backwards compatibility
                    # with the on-disk ``content_hash`` records persisted
                    # by earlier releases. Switching to SHA-256 would
                    # invalidate every existing dedup record and force a
                    # full re-ingest on first refresh after upgrade.
                    # Tracked for the next major version bump.
                    current_hash = _hashlib.md5(
                        combined.encode("utf-8", errors="replace"), usedforsecurity=False
                    ).hexdigest()
                    if current_hash == record.get("content_hash"):
                        job["skipped"].append(source_id)
                        continue
                    brain.ingest(docs)
                    job["reingested"].append(source_id)
                except Exception as exc:
                    job["errors"].append({"source": source_id, "error": str(exc)})
            completed = datetime.now(timezone.utc)
            job["status"] = "completed"
            job["phase"] = "done"
            job["completed_at"] = completed.isoformat()
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)

    background_tasks.add_task(_run_refresh)
    return {"job_id": job_id, "status": "processing"}


@router.post("/ingest")
async def ingest_data(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    req: Request,
):
    from axon import api as _api
    from axon.api_routes import metrics as _metrics

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
    _enforce_write_access(brain, "ingest")
    _metrics.record_ingest(
        project=_project_label(brain),
        surface=getattr(req.state, "surface", "api"),
    )
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

        # Capture reference once so repeated dict lookups can't KeyError if the
        # job is ever evicted from _api._jobs while the background task is running.
        job = _api._jobs.get(job_id)
        if job is None:
            return
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
            job["phase"] = "loading"
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
            job["files_total"] = len(docs)
            if docs:
                brain.ingest(docs, progress_callback=_make_progress_callback(job))
            job["status"] = "completed"
            job["phase"] = "completed"
            job["documents_ingested"] = len(docs)
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
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
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = datetime.now(timezone.utc).isoformat()
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
        getattr(brain, "_community_build_in_progress", False)
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
    from axon.api_routes import metrics as _metrics

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(request.project, brain)
    _enforce_write_access(brain, "ingest")
    _metrics.record_ingest(
        project=_project_label(brain, request.project),
        surface="api",
    )
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
        await asyncio.to_thread(brain.ingest, documents)
        _api._record_dedup(request.text, doc_id, project_key)
        return {"status": "success", "doc_id": doc_id, "chunks": len(documents)}
    except Exception as e:
        logger.error(f"Error adding text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add_texts")
async def add_texts(request: BatchTextIngestRequest):
    """Ingest a list of documents in a single embedding batch."""
    from axon import api as _api
    from axon.api_routes import metrics as _metrics

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    from axon.api_schemas import _compute_content_hash
    from axon.loaders import SmartTextLoader

    _enforce_project(request.project, brain)
    _enforce_write_access(brain, "ingest")
    _metrics.record_ingest(
        project=_project_label(brain, request.project),
        surface="api",
    )
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
            await asyncio.to_thread(brain.ingest, docs_to_ingest)
            for doc_id, text in pending_records:
                _api._record_dedup(text, doc_id, project_key)
        except Exception as e:
            logger.error(f"Error batch-ingesting texts: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    return results


@router.post("/ingest_url")
async def ingest_url(request: URLIngestRequest, req: Request):
    """Fetch an HTTP/HTTPS URL and ingest its text content."""
    from axon import api as _api
    from axon.api_routes import metrics as _metrics

    enforce_rate_limit(req, bucket="ingest_url", max_hits=20, window_seconds=60.0)
    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    from axon.loaders import URLLoader

    _enforce_project(request.project, brain)
    _enforce_write_access(brain, "ingest")
    _metrics.record_ingest(
        project=_project_label(brain, request.project),
        surface="api",
    )
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
        await asyncio.to_thread(brain.ingest, [doc])
        _api._record_dedup(doc["text"], doc["id"], project_key)
    except Exception as exc:
        logger.error(f"Error ingesting URL content: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ingested", "doc_id": doc["id"], "url": request.url}


@router.post("/ingest/upload")
async def ingest_upload(
    req: Request,
    files: list[UploadFile] = File(...),
    project: str | None = Form(None),
):
    """Accept multipart file uploads, load them through the normal file loaders, and ingest."""
    from axon import api as _api
    from axon.api_routes import metrics as _metrics

    enforce_rate_limit(req, bucket="ingest_upload", max_hits=30, window_seconds=60.0)
    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    _enforce_project(project, brain)
    _enforce_write_access(brain, "ingest")
    _metrics.record_ingest(
        project=_project_label(brain, project),
        surface="api",
    )
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required.")
    # Defence in depth: reject obviously oversized batches before any I/O.
    # Caps are real AxonConfig dataclass fields; fall back to the module
    # constants when the config is shaped differently (unit tests with mocks).
    _cfg_max_files = getattr(brain.config, "max_files_per_request", None)
    max_files: int = _cfg_max_files if isinstance(_cfg_max_files, int) else MAX_FILES_PER_REQUEST
    _cfg_max_bytes = getattr(brain.config, "max_upload_bytes", None)
    max_upload_bytes: int = _cfg_max_bytes if isinstance(_cfg_max_bytes, int) else MAX_UPLOAD_BYTES
    if len(files) > max_files:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Too many files in one request: {len(files)} > {max_files}. "
                "Split the upload into smaller batches."
            ),
        )
    from axon.loaders import DirectoryLoader

    loader_mgr = DirectoryLoader()
    docs_to_ingest: list[dict] = []
    results: list[dict] = []
    total_chunks = 0
    with tempfile.TemporaryDirectory(prefix="axon_upload_") as tmp_dir:
        temp_root = pathlib.Path(tmp_dir)
        used_names: set[str] = set()
        for index, upload in enumerate(files):
            safe_name = _normalise_uploaded_filename(upload.filename, index)
            candidate_name = safe_name
            stem = pathlib.Path(safe_name).stem
            suffix = pathlib.Path(safe_name).suffix
            counter = 1
            while candidate_name.lower() in used_names:
                candidate_name = f"{stem}_{counter}{suffix}"
                counter += 1
            used_names.add(candidate_name.lower())
            ext = pathlib.Path(candidate_name).suffix.lower()
            loader = loader_mgr.loaders.get(ext)
            if loader is None:
                results.append(
                    {
                        "filename": candidate_name,
                        "status": "unsupported",
                        "error": f"Unsupported file type: {ext or 'no extension'}",
                        "chunks": 0,
                    }
                )
                await upload.close()
                continue
            temp_path = temp_root / candidate_name
            # Stream upload to disk in chunks to avoid reading entire file into memory.
            # Per-file size cap is sourced from AxonConfig.max_upload_bytes
            # so deployments can tune it without a code change.
            total_written = 0
            try:
                with open(temp_path, "wb") as _out:
                    while True:
                        chunk = await upload.read(1024 * 1024)  # 1 MB
                        if not chunk:
                            break
                        _out.write(chunk)
                        total_written += len(chunk)
                        if total_written > max_upload_bytes:
                            await upload.close()
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"File '{candidate_name}' exceeds the allowed "
                                    f"size of {max_upload_bytes} bytes"
                                ),
                            )
                await upload.close()
            except HTTPException:
                # Let the HTTPException bubble up to the client
                raise
            except Exception as exc:
                try:
                    await upload.close()
                except Exception:
                    pass
                logger.error("Failed to save uploaded file %s: %s", candidate_name, exc)
                results.append(
                    {
                        "filename": candidate_name,
                        "status": "error",
                        "error": "failed to save uploaded file",
                        "chunks": 0,
                    }
                )
                continue
            try:
                docs = await asyncio.to_thread(loader.load, str(temp_path))
            except Exception as exc:
                logger.error("Failed to ingest uploaded file %s: %s", candidate_name, exc)
                results.append(
                    {
                        "filename": candidate_name,
                        "status": "error",
                        "error": str(exc),
                        "chunks": 0,
                    }
                )
                continue
            for doc in docs:
                doc["metadata"]["source"] = candidate_name
                if doc["metadata"].get("type") not in _PATH_ENRICHMENT_EXCLUDED_TYPES:
                    doc["text"] = f"[File Path: {candidate_name}]\n{doc['text']}"
            docs_to_ingest.extend(docs)
            total_chunks += len(docs)
            results.append(
                {
                    "filename": candidate_name,
                    "status": "ingested",
                    "error": None,
                    "chunks": len(docs),
                }
            )
        if not docs_to_ingest:
            raise HTTPException(status_code=400, detail="No supported files found in upload")
        try:
            await asyncio.to_thread(brain.ingest, docs_to_ingest)
        except Exception as exc:
            logger.error("Error ingesting uploaded file batch: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))
    return {
        "status": "success",
        "files": results,
        "ingested_files": sum(1 for item in results if item["status"] == "ingested"),
        "ingested_chunks": total_chunks,
    }


@router.post("/delete")
async def delete_documents(
    request: DeleteRequest,
    req: Request,
):
    from axon import api as _api
    from axon import governance as gov

    brain = _api.brain
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    rid = getattr(req.state, "request_id", "")
    surface = getattr(req.state, "surface", "api")
    project = getattr(brain, "_active_project", "default")
    try:
        _enforce_write_access(brain, "delete")
        existing = brain.vector_store.get_by_ids(request.doc_ids)
        existing_ids_set = {doc["id"] for doc in existing}
        existing_ids = [i for i in request.doc_ids if i in existing_ids_set]
        not_found = [i for i in request.doc_ids if i not in existing_ids_set]
        # Expand any not-found IDs that are source doc IDs (e.g. returned by
        # ingest_text) to their actual chunk IDs via the BM25 corpus, then
        # delete those expanded chunk IDs from both stores.
        if not_found and brain.bm25 is not None:
            not_found_set = set(not_found)
            expanded: list[str] = []
            resolved_sources: set[str] = set()
            for chunk in brain.bm25.corpus:
                src = chunk.get("metadata", {}).get("source", "")
                if src in not_found_set:
                    expanded.append(chunk["id"])
                    resolved_sources.add(src)
            if expanded:
                vs_source_docs = brain.vector_store.get_by_ids(expanded)
                vs_source_ids = [d["id"] for d in vs_source_docs]
                if vs_source_ids:
                    brain.vector_store.delete_by_ids(vs_source_ids)
                brain.bm25.delete_documents(expanded)
                existing_ids.extend(expanded)
                not_found = [i for i in not_found if i not in resolved_sources]
        if existing_ids:
            # Delete chunk IDs that were found directly (not from source expansion)
            direct_ids = [i for i in existing_ids if i in existing_ids_set]
            if direct_ids:
                brain.vector_store.delete_by_ids(direct_ids)
                if brain.bm25 is not None:
                    brain.bm25.delete_documents(direct_ids)
            if brain._entity_graph:
                brain._prune_entity_graph(set(existing_ids))
            from axon import api as _api

            _api._purge_dedup(existing_ids, project)
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
