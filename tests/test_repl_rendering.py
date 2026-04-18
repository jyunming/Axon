"""Tests for REPL markdown pre-processing and scroll target logic."""


# ---------------------------------------------------------------------------
# _fence_unfenced_code tests
# ---------------------------------------------------------------------------

from axon.repl import _fence_unfenced_code


class TestFenceUnfencedCode:
    """Verify that unfenced code blocks are wrapped in triple-backtick fences."""

    def test_already_fenced_unchanged(self):
        text = "Hello\n\n```python\ndef foo():\n    pass\n```\n\nDone."
        assert _fence_unfenced_code(text) == text

    def test_unfenced_python_function(self):
        text = "Here is a function:\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n\nThat's it."
        result = _fence_unfenced_code(text)
        assert "```python" in result
        assert "def factorial(n):" in result
        assert result.count("```") == 2  # open + close

    def test_unfenced_import_block(self):
        text = "Install these:\n\nimport os\nimport sys\nfrom pathlib import Path\n\nDone."
        result = _fence_unfenced_code(text)
        assert "```python" in result
        assert "import os" in result

    def test_unfenced_javascript(self):
        text = "Example:\n\nfunction greet(name) {\n    console.log('Hello ' + name);\n}\n\nDone."
        result = _fence_unfenced_code(text)
        assert "```javascript" in result

    def test_unfenced_rust(self):
        text = 'Code:\n\nfn main() {\n    println!("Hello");\n}\n\nEnd.'
        result = _fence_unfenced_code(text)
        assert "```rust" in result

    def test_single_code_line_not_fenced(self):
        """A single code-like line shouldn't be fenced (below 2-line threshold)."""
        text = "Use this:\n\nimport os\n\nThat's all."
        result = _fence_unfenced_code(text)
        assert "```" not in result

    def test_blockquote_code_not_fenced(self):
        """Code inside blockquotes should not be auto-fenced."""
        text = "> def foo():\n>     return 1\n\nDone."
        result = _fence_unfenced_code(text)
        assert "```" not in result

    def test_list_item_code_not_fenced(self):
        """Code inside list items should not be auto-fenced."""
        text = "- import os\n- import sys\n\nDone."
        result = _fence_unfenced_code(text)
        assert "```" not in result

    def test_mixed_prose_and_code(self):
        text = (
            "Here is how to do it:\n\n"
            "def hello():\n"
            "    print('world')\n\n"
            "And then call it:\n\n"
            "hello()\nprint('done')\n\n"
            "That's all."
        )
        result = _fence_unfenced_code(text)
        assert "```python" in result
        assert "def hello():" in result

    def test_indented_code_with_signals(self):
        """Indented lines with code signals should be fenced."""
        text = "Example:\n\n    x = 10\n    y = x + 5\n    print(y)\n\nResult: 15"
        result = _fence_unfenced_code(text)
        assert "```" in result

    def test_preserves_existing_fences_mixed(self):
        """Existing fenced blocks stay intact while unfenced blocks get fenced."""
        text = (
            "```python\ndef existing():\n    pass\n```\n\n"
            "And also:\n\n"
            "def unfenced():\n    return 42\n\nDone."
        )
        result = _fence_unfenced_code(text)
        # Original fence preserved
        assert result.startswith("```python\ndef existing():\n    pass\n```")
        # New fence added for unfenced block
        assert result.count("```") == 4  # 2 original + 2 new

    def test_empty_text(self):
        assert _fence_unfenced_code("") == ""

    def test_pure_prose(self):
        text = "This is just a paragraph of text.\n\nAnother paragraph here."
        assert _fence_unfenced_code(text) == text

    def test_decorator_starts_block(self):
        text = "Class:\n\n@dataclass\nclass Foo:\n    x: int = 0\n\nDone."
        result = _fence_unfenced_code(text)
        assert "```" in result
        assert "@dataclass" in result

    def test_sql_block(self):
        text = "Query:\n\nSELECT name, age\nFROM users\nWHERE age > 18;\n\nResults."
        result = _fence_unfenced_code(text)
        assert "```sql" in result

    def test_trailing_code_run(self):
        """Code at the end of text (no trailing prose) should still be fenced."""
        text = "Example:\n\ndef last():\n    return True"
        result = _fence_unfenced_code(text)
        assert "```python" in result
        assert result.rstrip().endswith("```")


# ---------------------------------------------------------------------------
# _scroll_target tests (via mock)
# ---------------------------------------------------------------------------


class TestScrollTarget:
    """Test scroll target Point calculation logic.

    The new algorithm places the cursor at the desired TOP of the viewport
    when scrolling up.  This forces prompt-toolkit's
    ``_scroll_when_linewrapping`` to clamp ``vertical_scroll`` to that line.
    """

    def test_auto_scroll_returns_last_line(self):
        """When offset=0, cursor should be at the last line."""
        from prompt_toolkit.data_structures import Point

        text = "line1\nline2\nline3\nline4"
        last_line = text.count("\n")
        assert last_line == 3

        # Simulates: offset=0 → Point at last line
        offset = 0
        if offset == 0:
            target = Point(x=0, y=last_line)
        else:
            target = Point(x=0, y=0)  # placeholder
        assert target == Point(x=0, y=3)

    def test_scroll_up_places_cursor_at_viewport_top(self):
        """When offset>0, cursor is at topmost_visible - offset."""
        from prompt_toolkit.data_structures import Point

        # 51 lines (0-50), window height 39
        text = "\n".join(f"line{i}" for i in range(51))
        last_line = text.count("\n")  # 50
        wh = 39
        offset = 8

        topmost = max(0, last_line + 1 - wh)  # 51 - 39 = 12
        target_top = max(0, topmost - offset)  # 12 - 8 = 4
        target = Point(x=0, y=target_top)
        assert target == Point(x=0, y=4)

    def test_offset_larger_than_topmost(self):
        """Offset larger than topmost visible should clamp to line 0."""
        from prompt_toolkit.data_structures import Point

        text = "\n".join(f"line{i}" for i in range(51))
        last_line = text.count("\n")  # 50
        wh = 39
        offset = 20  # bigger than topmost(12)

        topmost = max(0, last_line + 1 - wh)  # 12
        target_top = max(0, topmost - offset)  # max(0, -8) = 0
        target = Point(x=0, y=target_top)
        assert target == Point(x=0, y=0)

    def test_empty_text(self):
        """Empty text should return Point(0, 0)."""
        from prompt_toolkit.data_structures import Point

        text = ""
        last_line = max(0, text.count("\n"))
        target = Point(x=0, y=last_line)
        assert target == Point(x=0, y=0)

    def test_small_content_no_scroll_needed(self):
        """When content fits in window, offset still works (clamps to 0)."""
        from prompt_toolkit.data_structures import Point

        text = "line1\nline2\nline3"  # 3 lines
        last_line = text.count("\n")  # 2
        wh = 39
        offset = 4

        topmost = max(0, last_line + 1 - wh)  # max(0, 3-39) = 0
        target_top = max(0, topmost - offset)  # max(0, -4) = 0
        target = Point(x=0, y=target_top)
        assert target == Point(x=0, y=0)


# ---------------------------------------------------------------------------
# Streaming chunk helper tests
# ---------------------------------------------------------------------------


class TestStreamingHelpers:
    """Verify the streaming chunk management functions."""

    def _make_env(self):
        """Create a minimal environment mimicking the REPL closures."""
        import threading

        chunks = []
        lock = threading.Lock()
        view = {"offset": 0, "sticky_bottom": True}
        invalidate_count = [0]

        class FakeApp:
            def invalidate(self):
                invalidate_count[0] += 1

        app = FakeApp()
        streaming_idx = [None]
        last_inv = [0.0]

        def start_streaming():
            with lock:
                chunks.append("")
                streaming_idx[0] = len(chunks) - 1
            if view["sticky_bottom"]:
                view["offset"] = 0

        def stream_token(token):
            import time

            if streaming_idx[0] is None:
                return
            indented = token.replace("\n", "\n  ")
            with lock:
                chunks[streaming_idx[0]] += indented
            if view["sticky_bottom"]:
                view["offset"] = 0
            now = time.monotonic()
            if now - last_inv[0] > 0.066:
                last_inv[0] = now
                app.invalidate()

        def finish_streaming(rendered):
            if streaming_idx[0] is not None:
                with lock:
                    chunks[streaming_idx[0]] = rendered
                streaming_idx[0] = None
            if view["sticky_bottom"]:
                view["offset"] = 0
            app.invalidate()

        return (
            chunks,
            streaming_idx,
            invalidate_count,
            start_streaming,
            stream_token,
            finish_streaming,
            view,
        )

    def test_start_streaming_creates_placeholder(self):
        chunks, idx, _, start, _, _, _ = self._make_env()
        assert len(chunks) == 0
        start()
        assert len(chunks) == 1
        assert chunks[0] == ""
        assert idx[0] == 0

    def test_stream_token_appends_to_chunk(self):
        chunks, _, _, start, token, _, _ = self._make_env()
        start()
        token("Hello")
        token(" world")
        assert chunks[0] == "Hello world"

    def test_stream_token_indents_newlines(self):
        chunks, _, _, start, token, _, _ = self._make_env()
        start()
        token("line1\nline2\nline3")
        assert chunks[0] == "line1\n  line2\n  line3"

    def test_finish_streaming_replaces_chunk(self):
        chunks, idx, _, start, token, finish, _ = self._make_env()
        start()
        token("raw tokens here")
        assert "raw tokens" in chunks[0]
        finish("[bold]Rendered markdown[/bold]")
        assert chunks[0] == "[bold]Rendered markdown[/bold]"
        assert idx[0] is None

    def test_stream_token_noop_without_start(self):
        chunks, _, _, _, token, _, _ = self._make_env()
        token("orphan token")
        assert len(chunks) == 0

    def test_finish_streaming_noop_without_start(self):
        chunks, idx, inv, _, _, finish, _ = self._make_env()
        finish("rendered")
        assert len(chunks) == 0
        assert idx[0] is None

    def test_multiple_streaming_sessions(self):
        chunks, idx, _, start, token, finish, _ = self._make_env()
        # First session
        start()
        token("first")
        finish("FIRST_RENDERED")
        assert chunks[0] == "FIRST_RENDERED"
        # Second session
        start()
        assert idx[0] == 1
        token("second")
        finish("SECOND_RENDERED")
        assert chunks[1] == "SECOND_RENDERED"

    def test_invalidate_throttling(self):
        """invalidate should not be called on every token when tokens arrive fast."""

        chunks, _, inv, start, token, _, _ = self._make_env()
        start()
        inv[0] = 0
        # Fire many tokens rapidly
        for i in range(50):
            token(f"t{i}")
        # With 50 rapid calls, throttling at 66ms should limit invalidations
        # At minimum 1 (the first call), at most a few
        assert inv[0] >= 1
        assert inv[0] < 50

    # ── Sticky-bottom scroll persistence tests ────────────────────────

    def test_sticky_bottom_default_true(self):
        """View starts with sticky_bottom=True."""
        *_, view = self._make_env()
        assert view["sticky_bottom"] is True

    def test_scroll_preserved_when_not_sticky(self):
        """When sticky_bottom=False, streaming does NOT reset offset."""
        _, _, _, start, token, finish, view = self._make_env()
        # Simulate user scrolling up
        view["sticky_bottom"] = False
        view["offset"] = 16
        # Streaming should NOT touch offset
        start()
        assert view["offset"] == 16
        token("hello")
        assert view["offset"] == 16
        token(" world")
        assert view["offset"] == 16
        finish("rendered output")
        assert view["offset"] == 16

    def test_offset_resets_when_sticky(self):
        """When sticky_bottom=True, streaming resets offset to 0."""
        _, _, _, start, token, finish, view = self._make_env()
        view["sticky_bottom"] = True
        view["offset"] = 5  # leftover from earlier
        start()
        assert view["offset"] == 0
        view["offset"] = 3
        token("tok")
        assert view["offset"] == 0
        view["offset"] = 7
        finish("done")
        assert view["offset"] == 0


# ---------------------------------------------------------------------------
# _normalize_bullets tests
# ---------------------------------------------------------------------------

from axon.repl import _normalize_bullets


class TestNormalizeBullets:
    """Verify that • bullet characters are converted to markdown - lists."""

    def test_basic_bullet_conversion(self):
        text = "Items:\n• First\n• Second\n• Third"
        result = _normalize_bullets(text)
        assert result == "Items:\n- First\n- Second\n- Third"

    def test_indented_bullets(self):
        text = "  • Sub-item 1\n  • Sub-item 2"
        result = _normalize_bullets(text)
        assert result == "  - Sub-item 1\n  - Sub-item 2"

    def test_no_bullets_unchanged(self):
        text = "Regular text without bullets."
        assert _normalize_bullets(text) == text

    def test_bullet_with_space(self):
        text = "• I_0 is the intensity.\n• R is the distance."
        result = _normalize_bullets(text)
        assert result == "- I_0 is the intensity.\n- R is the distance."

    def test_mixed_bullets_and_prose(self):
        text = "In this equation:\n• I_0 is intensity\nAnd also:\n• R is distance"
        result = _normalize_bullets(text)
        assert "- I_0 is intensity" in result
        assert "- R is distance" in result
        assert "•" not in result


# ---------------------------------------------------------------------------
# _fence_math_formulas tests
# ---------------------------------------------------------------------------

from axon.repl import _fence_math_formulas


class TestFenceMathFormulas:
    """Verify standalone math formulas are wrapped in code blocks with Unicode."""

    def test_rayleigh_formula(self):
        """The Rayleigh scattering formula from the user's real output."""
        text = (
            "The formula is:\n\n"
            "I = I_0 * ((1 + cos^2(theta)) / (2 * R^2)) * "
            "((2 * pi^2) / lambda^4) * ((n^2 - 1) / (n^2 + 2))^2 * V^2\n\n"
            "Where I is intensity."
        )
        result = _fence_math_formulas(text)
        assert "```" in result
        # Unicode conversions should be applied
        assert "I₀" in result  # subscript
        assert "cos²" in result  # superscript
        assert "θ" in result  # Greek
        assert "·" in result  # multiplication dot
        assert "π" in result  # pi
        assert "λ" in result  # lambda
        lines = result.split("\n")
        fence_count = sum(1 for ln in lines if ln.strip() == "```")
        assert fence_count == 2

    def test_simple_equation(self):
        """E = mc^2 should be fenced with Unicode superscript."""
        text = "Einstein's equation:\n\nE = m * c^2\n\nWhere E is energy."
        result = _fence_math_formulas(text)
        assert "```" in result
        assert "c²" in result
        assert "·" in result

    def test_simple_assignment_not_fenced(self):
        """A simple assignment like x = 5 should NOT be fenced."""
        text = "Set the value:\n\nx = 5\n\nDone."
        result = _fence_math_formulas(text)
        assert "```" not in result

    def test_prose_with_equals_not_fenced(self):
        """Prose sentences with = should not be fenced."""
        text = "The result = 42 in all cases."
        result = _fence_math_formulas(text)
        assert "```" not in result

    def test_already_fenced_formula_unchanged(self):
        """Formulas inside existing fences should not be double-fenced."""
        text = "```\nE = m * c^2\n```"
        result = _fence_math_formulas(text)
        assert result.count("```") == 2  # only original fences

    def test_quadratic_formula(self):
        """Quadratic formula with compact multiplication (4*a*c)."""
        text = "x = (-b + sqrt(b^2 - 4*a*c)) / (2*a)"
        result = _fence_math_formulas(text)
        assert "```" in result
        assert "4·a·c" in result  # compact mul converted
        assert "b²" in result

    def test_trig_formula(self):
        """Formula with trig functions and Greek letters."""
        text = "y = A * sin(omega * t + phi)"
        result = _fence_math_formulas(text)
        assert "```" in result
        assert "ω" in result  # omega → ω
        assert "φ" in result  # phi → φ

    def test_list_item_not_fenced(self):
        """Formula-like text inside list items should not be fenced."""
        text = "- x = A * sin(theta)"
        result = _fence_math_formulas(text)
        assert result.count("```") == 0


# ---------------------------------------------------------------------------
# _mathify tests — ASCII math → Unicode scientific symbols
# ---------------------------------------------------------------------------

from axon.repl import _mathify


class TestMathify:
    """Verify ASCII→Unicode math conversion for terminal display."""

    def test_superscript_digits(self):
        assert _mathify("x^2") == "x²"
        assert _mathify("n^3") == "n³"

    def test_subscript_digits(self):
        assert _mathify("I_0") == "I₀"
        assert _mathify("R_1") == "R₁"

    def test_greek_letters(self):
        assert _mathify("theta") == "θ"
        assert _mathify("lambda") == "λ"
        assert _mathify("pi") == "π"
        assert _mathify("omega") == "ω"
        assert _mathify("Sigma") == "Σ"

    def test_multiplication_with_spaces(self):
        assert _mathify("2 * R") == "2 · R"

    def test_multiplication_compact(self):
        assert _mathify("4*a*c") == "4·a·c"

    def test_comparison_operators(self):
        assert _mathify("x >= 0") == "x ≥ 0"
        assert _mathify("a <= b") == "a ≤ b"
        assert _mathify("a != b") == "a ≠ b"

    def test_rayleigh_full(self):
        """Full Rayleigh formula conversion."""
        inp = "I = I_0 * ((1 + cos^2(theta)) / (2 * R^2))"
        result = _mathify(inp)
        assert "I₀" in result
        assert "cos²" in result
        assert "θ" in result
        assert "R²" in result
        assert "·" in result

    def test_no_conversion_in_prose(self):
        """Words that happen to contain Greek substrings stay unchanged."""
        # 'pi' at word boundary should convert, but 'picture' should not.
        assert _mathify("picture") == "picture"
        assert _mathify("capital") == "capital"

    def test_partial_superscript_fallback(self):
        """Superscript chars not in the map should leave group unchanged."""
        # The char map covers a-z digits; chars like '.' are not mapped.
        # ^{.} should stay as ^{.} since '.' has no superscript form.
        result = _mathify("x^{.}")
        assert result == "x^{.}"  # not converted

    def test_infinity_keyword(self):
        assert _mathify("x -> infinity") == "x -> ∞"


# ---------------------------------------------------------------------------
# Inline math symbols tests
# ---------------------------------------------------------------------------

from axon.repl import _inline_math_symbols


class TestInlineMathSymbols:
    """Verify Greek + subscript conversion in prose lines."""

    def test_greek_in_bullet(self):
        text = "- theta is the scattering angle."
        assert "θ" in _inline_math_symbols(text)

    def test_subscript_in_bullet(self):
        text = "- I_0 is the intensity."
        assert "I₀" in _inline_math_symbols(text)

    def test_lambda_in_bullet(self):
        text = "- lambda is the wavelength."
        assert "λ" in _inline_math_symbols(text)

    def test_no_conversion_inside_fence(self):
        text = "```\ntheta = 3.14\nI_0 = 100\n```"
        result = _inline_math_symbols(text)
        assert "θ" not in result
        assert "I₀" not in result

    def test_plain_prose_unchanged(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert _inline_math_symbols(text) == text

    def test_mixed_prose_and_math(self):
        text = "Where R is the distance and theta is the angle."
        result = _inline_math_symbols(text)
        assert "θ" in result
        assert "R is the distance" in result

    def test_multiple_subscripts(self):
        text = "Variables x_1, x_2, and x_3 are used."
        result = _inline_math_symbols(text)
        assert "x₁" in result
        assert "x₂" in result
        assert "x₃" in result


# ---------------------------------------------------------------------------
# Scroll event handler tests
# ---------------------------------------------------------------------------


class TestScrollEventHandlers:
    """Verify scroll offset management for mouse and keyboard scroll events."""

    def test_scroll_up_increments_offset(self):
        view = {"offset": 0, "sticky_bottom": True}
        # Simulate scroll-up: offset += 3, sticky_bottom = False
        view["offset"] = min(view["offset"] + 3, 100)
        view["sticky_bottom"] = False
        assert view["offset"] == 3
        assert view["sticky_bottom"] is False

    def test_scroll_down_decrements_offset(self):
        view = {"offset": 12, "sticky_bottom": False}
        view["offset"] = max(0, view["offset"] - 3)
        if view["offset"] == 0:
            view["sticky_bottom"] = True
        assert view["offset"] == 9
        assert view["sticky_bottom"] is False

    def test_scroll_down_to_bottom_engages_sticky(self):
        view = {"offset": 2, "sticky_bottom": False}
        view["offset"] = max(0, view["offset"] - 3)
        if view["offset"] == 0:
            view["sticky_bottom"] = True
        assert view["offset"] == 0
        assert view["sticky_bottom"] is True

    def test_scroll_offset_clamped_to_max(self):
        """Offset should not exceed total line count."""
        text = "line1\nline2\nline3"
        max_offset = max(0, text.count("\n"))  # 2
        view = {"offset": 0, "sticky_bottom": True}
        view["offset"] = min(view["offset"] + 8, max_offset)
        assert view["offset"] == 2  # clamped to max

    def test_ctrl_scroll_single_line(self):
        """Ctrl+Up/Down should scroll by 1 line."""
        view = {"offset": 5, "sticky_bottom": False}
        # Ctrl+Up
        view["offset"] = min(view["offset"] + 1, 100)
        assert view["offset"] == 6
        # Ctrl+Down
        view["offset"] = max(0, view["offset"] - 1)
        assert view["offset"] == 5

    def test_page_up_scrolls_by_8(self):
        view = {"offset": 0, "sticky_bottom": True}
        view["offset"] = min(view["offset"] + 8, 100)
        view["sticky_bottom"] = False
        assert view["offset"] == 8
        assert view["sticky_bottom"] is False


# ---------------------------------------------------------------------------
# _ScrollableConversation mouse handler tests (cross-platform scroll)
# ---------------------------------------------------------------------------

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType


class TestScrollableConversationMouse:
    """Verify that the custom FormattedTextControl subclass handles
    SCROLL_UP / SCROLL_DOWN mouse events on all platforms (Windows
    via WindowsMouseEvent, VT100 via Vt100MouseEvent, both route
    through UIControl.mouse_handler)."""

    @staticmethod
    def _make_event(event_type):
        """Create a minimal MouseEvent for testing."""
        return MouseEvent(
            position=Point(x=0, y=0),
            event_type=event_type,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )

    @staticmethod
    def _make_control(view, max_offset=100):
        """Build a _ScrollableConversation-like control with closure vars."""
        from prompt_toolkit.layout.controls import FormattedTextControl

        class _ScrollableConversation(FormattedTextControl):
            def mouse_handler(self, mouse_event):
                if mouse_event.event_type == MouseEventType.SCROLL_UP:
                    view["offset"] = min(view["offset"] + 3, max_offset)
                    view["sticky_bottom"] = False
                    return None
                if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                    view["offset"] = max(0, view["offset"] - 3)
                    if view["offset"] == 0:
                        view["sticky_bottom"] = True
                    return None
                return super().mouse_handler(mouse_event)

        return _ScrollableConversation("")

    def test_scroll_up_updates_offset(self):
        view = {"offset": 0, "sticky_bottom": True}
        ctrl = self._make_control(view)
        result = ctrl.mouse_handler(self._make_event(MouseEventType.SCROLL_UP))
        assert result is None  # consumed
        assert view["offset"] == 3
        assert view["sticky_bottom"] is False

    def test_scroll_down_updates_offset(self):
        view = {"offset": 9, "sticky_bottom": False}
        ctrl = self._make_control(view)
        result = ctrl.mouse_handler(self._make_event(MouseEventType.SCROLL_DOWN))
        assert result is None
        assert view["offset"] == 6
        assert view["sticky_bottom"] is False  # not at bottom yet

    def test_scroll_down_to_bottom_sets_sticky(self):
        view = {"offset": 2, "sticky_bottom": False}
        ctrl = self._make_control(view)
        result = ctrl.mouse_handler(self._make_event(MouseEventType.SCROLL_DOWN))
        assert result is None
        assert view["offset"] == 0
        assert view["sticky_bottom"] is True

    def test_scroll_up_clamped_to_max(self):
        view = {"offset": 8, "sticky_bottom": False}
        ctrl = self._make_control(view, max_offset=10)
        result = ctrl.mouse_handler(self._make_event(MouseEventType.SCROLL_UP))
        assert result is None
        assert view["offset"] == 10  # clamped, not 11

    def test_non_scroll_event_passes_through(self):
        view = {"offset": 5, "sticky_bottom": False}
        ctrl = self._make_control(view)
        result = ctrl.mouse_handler(self._make_event(MouseEventType.MOUSE_DOWN))
        assert result == NotImplemented
        assert view["offset"] == 5  # unchanged

    def test_multiple_scroll_up_accumulates(self):
        view = {"offset": 0, "sticky_bottom": True}
        ctrl = self._make_control(view)
        for _ in range(4):
            ctrl.mouse_handler(self._make_event(MouseEventType.SCROLL_UP))
        assert view["offset"] == 12  # 4 * 3
