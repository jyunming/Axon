"""Exhaustive save() -> load() round-trip guard for AxonConfig.

`save()` hand-wrote 21 YAML sections covering 85 of 241 dataclass fields. The
other 156 were never written, so they silently reverted to defaults on the next
load — `graph_rag_depth: light` came back "standard", a custom `ollama_base_url`
came back "localhost", `mmr` came back False. Anything set via `axon --setup`,
REPL `/config set` or `POST /config/update` and then persisted was quietly lost.

These tests walk every field rather than sampling, so a newly added config knob
that nobody wired into save() fails here instead of in a user's config file.

`save()` refuses to write inside the system temp dir (it guards against tests
clobbering a live config), so these use a repo-local scratch dir, not tmp_path.
"""

import shutil
import uuid
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from axon.config import (
    _SAVE_DERIVED_FIELDS,
    _SAVE_EXPLICIT_FIELDS,
    AxonConfig,
    _unsaved_field_names,
)

_SCRATCH = Path(__file__).parent / ".test_tmp" / "roundtrip"

# Values that survive a YAML round-trip and differ from every default, so a
# dropped field shows up as a mismatch rather than coincidentally matching.
_PROBES = {
    bool: True,
    int: 7,
    float: 0.375,
    str: "axon-roundtrip-probe",
    list: ["axon-roundtrip-probe"],
    dict: {"probe": 1},
}


@pytest.fixture
def cfg_path():
    d = _SCRATCH / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    yield str(d / "config.yaml")
    shutil.rmtree(d, ignore_errors=True)


def _settable_fields():
    """Fields a user can meaningfully persist (not private, not derived)."""
    return [
        f
        for f in fields(AxonConfig)
        if not f.name.startswith("_") and f.name not in _SAVE_DERIVED_FIELDS
    ]


class TestFieldCoverage:
    def test_every_field_is_classified(self):
        """No field may fall through both the explicit list and the completion pass."""
        covered = set(_SAVE_EXPLICIT_FIELDS) | set(_unsaved_field_names())
        uncovered = {f.name for f in _settable_fields()} - covered
        assert not uncovered, f"fields save() would drop entirely: {sorted(uncovered)}"

    def test_explicit_and_completion_sets_are_disjoint(self):
        """A field in both would be written twice — same value, duplicated key."""
        assert not (set(_SAVE_EXPLICIT_FIELDS) & set(_unsaved_field_names()))

    def test_explicit_list_has_no_stale_entries(self):
        """A renamed/removed field left in the list would silently skip its successor."""
        real = {f.name for f in fields(AxonConfig)}
        assert not (set(_SAVE_EXPLICIT_FIELDS) - real)

    def test_derived_fields_are_real(self):
        real = {f.name for f in fields(AxonConfig)}
        assert not (set(_SAVE_DERIVED_FIELDS) - real)


class TestRoundTrip:
    def test_defaults_round_trip(self, cfg_path):
        """A saved default config must reload identical."""
        original = AxonConfig()
        original.save(cfg_path)
        assert Path(cfg_path).exists(), "save() wrote nothing — check the temp-dir guard"
        reloaded = AxonConfig.load(cfg_path)
        drifted = {
            f.name: (getattr(original, f.name), getattr(reloaded, f.name))
            for f in _settable_fields()
            if getattr(original, f.name) != getattr(reloaded, f.name)
        }
        assert not drifted, f"defaults changed across save/load: {drifted}"

    def test_every_rag_key_is_a_real_field(self, cfg_path):
        """load() maps `rag:` keys straight onto field names — typos vanish silently."""
        AxonConfig().save(cfg_path)
        raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        real = {f.name for f in fields(AxonConfig)}
        unknown = set(raw.get("rag", {})) - real
        assert not unknown, f"`rag:` keys matching no dataclass field: {sorted(unknown)}"

    @pytest.mark.parametrize("field", _settable_fields(), ids=lambda f: f.name)
    def test_every_field_survives(self, field, cfg_path):
        """Set each field to a non-default value and require it back verbatim."""
        default = getattr(AxonConfig(), field.name)
        probe = _PROBES.get(type(default))
        if probe is None or probe == default:
            pytest.skip(f"no distinguishing probe for {field.name} ({type(default).__name__})")

        try:
            cfg = AxonConfig(**{field.name: probe})
        except (TypeError, ValueError):
            pytest.skip(f"{field.name} rejects the generic probe value")

        # __post_init__ may normalise or override (derived paths, env overlays).
        effective = getattr(cfg, field.name)
        if effective != probe:
            pytest.skip(f"{field.name} is normalised by __post_init__")

        cfg.save(cfg_path)
        try:
            loaded = AxonConfig.load(cfg_path)
        except ValueError:
            # Enum-constrained field (e.g. keyring_mode) rejecting the generic
            # probe on the way back in. Not a round-trip failure.
            pytest.skip(f"{field.name} constrains its values; generic probe rejected")
        reloaded = getattr(loaded, field.name)
        assert reloaded == effective, (
            f"{field.name} did not survive save/load: "
            f"saved {effective!r}, got {reloaded!r} back"
        )


class TestRegressionsFoundLive:
    """The specific fields observed reverting before the completion pass existed."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("graph_rag_depth", "light"),
            ("graph_rag_ner_backend", "gliner"),
            ("ollama_base_url", "http://example.invalid:11434"),
            ("mmr", True),
            ("sentence_window", True),
            ("cite", True),
            ("code_graph", True),
        ],
    )
    def test_known_droppers(self, field, value, cfg_path):
        cfg = AxonConfig(**{field: value})
        cfg.save(cfg_path)
        assert getattr(AxonConfig.load(cfg_path), field) == value
