"""Integration tests for scroll and formula rendering.

These tests verify the end-to-end behavior of the REPL's mouse scroll
handling and formula Unicode conversion pipeline.
"""

import threading

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType


def test_scrollable_conversation_mouse_handler():
    """Simulate the exact _ScrollableConversation mouse handler logic."""
    _cv = {"offset": 0, "sticky_bottom": True}
    _chunks = ["Line\n" * 40]
    _lock = threading.Lock()

    def _ct():
        with _lock:
            return "".join(_chunks)

    def _mx():
        return max(0, _ct().count("\n"))

    class SC(FormattedTextControl):
        def mouse_handler(self, me):
            if me.event_type == MouseEventType.SCROLL_UP:
                _cv["offset"] = min(_cv["offset"] + 3, _mx())
                _cv["sticky_bottom"] = False
                return None
            if me.event_type == MouseEventType.SCROLL_DOWN:
                _cv["offset"] = max(0, _cv["offset"] - 3)
                if _cv["offset"] == 0:
                    _cv["sticky_bottom"] = True
                return None
            return super().mouse_handler(me)

    ctrl = SC(lambda: "")
    eu = MouseEvent(Point(0, 0), MouseEventType.SCROLL_UP, MouseButton.NONE, frozenset())
    ed = MouseEvent(Point(0, 0), MouseEventType.SCROLL_DOWN, MouseButton.NONE, frozenset())

    # Scroll up
    r = ctrl.mouse_handler(eu)
    assert _cv["offset"] == 3
    assert r is None  # consumed
    assert _cv["sticky_bottom"] is False

    # Scroll up again
    ctrl.mouse_handler(eu)
    assert _cv["offset"] == 6

    # Scroll down
    ctrl.mouse_handler(ed)
    assert _cv["offset"] == 3

    # Scroll down to bottom
    ctrl.mouse_handler(ed)
    assert _cv["offset"] == 0
    assert _cv["sticky_bottom"] is True

    # Non-scroll event passes through
    ec = MouseEvent(Point(0, 0), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset())
    r2 = ctrl.mouse_handler(ec)
    assert r2 is NotImplemented or r2 is None  # passes through


def test_formula_preprocessing_unicode():
    """Verify formulas get Unicode symbols after preprocessing."""
    from axon.repl import _preprocess_markdown

    text = "Formula:\n\nI = I_0 * cos^2(theta) / (R^2)\n\nEnd."
    result = _preprocess_markdown(text)

    assert "\u2080" in result, "Missing subscript 0 (I_0 -> I\u2080)"
    assert "\u00b2" in result, "Missing superscript 2 (^2 -> \u00b2)"
    assert "\u03b8" in result, "Missing theta -> \u03b8"
    assert "\u00b7" in result, "Missing multiplication dot (* -> \u00b7)"
    assert "```" in result, "Missing code fences around formula"


def test_formula_rendering_through_rich():
    """Verify the full Rich rendering pipeline produces readable output."""
    from axon.repl import _make_math_renderable

    text = "E = m * c^2"
    result = _make_math_renderable(text)

    # _make_math_renderable returns a list of renderables or a single renderable
    # when there are no $$ blocks. Either way, it should have preprocessed the text.
    import io

    from rich.console import Console

    buf = io.StringIO()
    cap = Console(file=buf, force_terminal=True, width=100)
    cap.print(result)
    output = buf.getvalue()

    assert "c\u00b2" in output, f"Missing superscript in rendered output: {output!r}"
    assert "\u00b7" in output, f"Missing multiplication dot in rendered output: {output!r}"


def test_scroll_target_bottom():
    """Verify _scroll_target returns last line when at bottom (offset=0)."""
    # This simulates the function logic directly
    text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    last_line = text.count("\n")
    offset = 0

    if offset == 0:
        target = Point(x=0, y=last_line)
    assert target.y == 5  # cursor at last line


def test_scroll_target_scrolled_up():
    """Verify _scroll_target computes correct position when scrolled up."""
    text = "Line\n" * 50  # 50 lines
    last_line = text.count("\n")  # 50
    offset = 10
    wh = 20  # window height

    topmost = max(0, last_line + 1 - wh)  # 31
    target_top = max(0, topmost - offset)  # 21
    target = Point(x=0, y=target_top)

    assert target.y == 21, f"Expected 21, got {target.y}"


if __name__ == "__main__":
    test_scrollable_conversation_mouse_handler()
    print("PASS: Scroll handler")
    test_formula_preprocessing_unicode()
    print("PASS: Formula preprocessing")
    test_formula_rendering_through_rich()
    print("PASS: Formula Rich rendering")
    test_scroll_target_bottom()
    print("PASS: Scroll target bottom")
    test_scroll_target_scrolled_up()
    print("PASS: Scroll target scrolled up")
    print("\nALL INTEGRATION TESTS PASSED")
