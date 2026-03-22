"""API route sub-package — each module registers an APIRouter."""
from fastapi import HTTPException


def enforce_project(requested: str | None, brain) -> None:
    """Raise 409 if the caller requested a project that is not the active one.

    The brain is a singleton that serves one project at a time.  Silently
    ignoring a mismatched ``project`` field would read/write the wrong corpus.
    Callers should switch first via POST /project/switch.
    """
    if not requested:
        return
    active = getattr(brain, "_active_project", "default") or "default"
    if requested != active:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Brain is serving project '{active}', not '{requested}'. "
                "Use POST /project/switch to change the active project."
            ),
        )
