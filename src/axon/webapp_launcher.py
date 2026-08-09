"""Console-script entry point for the deprecated Streamlit UI (``axon-ui``).

This module deliberately imports **nothing** from ``axon.webapp`` and does not
import ``streamlit``. ``webapp.py`` is a Streamlit *script*: it calls
``st.set_page_config()`` and renders the whole sidebar at module scope, so
importing it without Streamlit installed raises ``NameError: name 'st' is not
defined`` before any of its own error handling can run.

Keeping the launcher separate means:

* ``axon-ui`` prints an actionable install hint instead of a raw traceback when
  Streamlit is missing — which is now the common case, since Streamlit was
  dropped from the ``[starter]`` and ``[all]`` extras;
* the deprecation notice is testable in CI, where Streamlit is not installed.

.. deprecated::
    The Streamlit UI is superseded by the native web GUI served at ``/gui/`` by
    ``axon-api``. This entry point still works but will be removed in a future
    release.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings

DEPRECATION_NOTICE = (
    "DEPRECATED: the Streamlit UI (axon-ui) is deprecated and will be removed "
    "in a future release.\n"
    "Use the native web GUI instead — it is served by the API server:\n"
    "\n"
    "    axon-api            # then open http://localhost:8000/gui/\n"
    "\n"
    "The native GUI is the maintained surface and exposes more capability "
    "(graph explorer, governance console, knowledge base).\n"
)

MISSING_STREAMLIT_NOTICE = (
    "ERROR: streamlit is not installed.\n"
    "The Streamlit UI is deprecated and is no longer part of the [starter] or "
    "[all] extras.\n"
    "\n"
    "Preferred — use the native web GUI, which needs no extra dependency:\n"
    "\n"
    "    axon-api            # then open http://localhost:8000/gui/\n"
    "\n"
    "Or install the deprecated UI explicitly:\n"
    "\n"
    '    pip install "axon-rag[ui]"\n'
)


def _streamlit_available() -> bool:
    """Return True when ``streamlit`` can be imported.

    Uses :func:`importlib.util.find_spec` rather than a real import so the check
    stays cheap and has no side effects.
    """
    from importlib.util import find_spec

    try:
        return find_spec("streamlit") is not None
    except (ImportError, ValueError):
        # ValueError: a test monkeypatched sys.modules["streamlit"] = None.
        return False


def webapp_script_path() -> str:
    """Absolute path to the Streamlit script that ``streamlit run`` executes."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp.py")


def main_ui() -> None:
    """Entry point for the ``axon-ui`` command.

    Announces the deprecation, then launches the Streamlit script if Streamlit
    is installed. Exits with status 1 and an actionable hint when it is not.
    """
    warnings.warn(
        "axon-ui (Streamlit) is deprecated; use the native web GUI at "
        "http://localhost:8000/gui/ served by axon-api.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(DEPRECATION_NOTICE, file=sys.stderr)

    if not _streamlit_available():
        print(MISSING_STREAMLIT_NOTICE, file=sys.stderr)
        sys.exit(1)

    subprocess.run(["streamlit", "run", webapp_script_path()])
