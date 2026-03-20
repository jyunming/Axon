import json as _json
import os
from datetime import datetime as _dt
from datetime import timezone as _tz
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axon.main import AxonBrain

_SESSIONS_DIR = os.path.join(os.path.expanduser("~"), ".axon", "sessions")


def _sessions_dir(project: str | None = None) -> str:
    """Return the sessions directory for *project*, or the global fallback."""
    if project and project != "default":
        from axon.projects import project_sessions_path

        d = project_sessions_path(project)
    else:
        d = _SESSIONS_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _new_session(brain: "AxonBrain") -> dict:
    return {
        "id": _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S%f")[:-3],
        "started_at": _dt.now(_tz.utc).isoformat(),
        "provider": brain.config.llm_provider,
        "model": brain.config.llm_model,
        "project": getattr(brain, "_active_project", "default"),
        "history": [],
    }


def _session_path(session_id: str, project: str | None = None) -> str:
    return os.path.join(_sessions_dir(project), f"session_{session_id}.json")


def _save_session(session: dict) -> None:
    try:
        project = session.get("project")
        with open(_session_path(session["id"], project), "w", encoding="utf-8") as f:
            _json.dump(session, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _list_sessions(limit: int = 20, project: str | None = None) -> list:
    d = _sessions_dir(project)
    files = sorted(
        [f for f in os.listdir(d) if f.startswith("session_") and f.endswith(".json")],
        reverse=True,
    )[:limit]
    sessions = []
    for fn in files:
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                s = _json.load(f)
            sessions.append(s)
        except Exception:
            pass
    return sessions


def _load_session(session_id: str, project: str | None = None) -> dict | None:
    p = _session_path(session_id, project)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def _print_sessions(sessions: list) -> None:
    if not sessions:
        print("  (no saved sessions)")
        return
    print(f"\n  {'ID':<18}  {'Model':<30}  {'Turns':<6}  Started")
    print(f"  {'─'*18}  {'─'*30}  {'─'*6}  {'─'*20}")
    for s in sessions:
        turns = len(s.get("history", [])) // 2
        ts = s.get("started_at", "")[:16].replace("T", " ")
        model = f"{s.get('provider','?')}/{s.get('model','?')}"
        print(f"  {s['id']:<18}  {model:<30}  {turns:<6}  {ts}")
    print()
