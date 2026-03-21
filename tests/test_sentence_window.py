import axon.sentence_window as sw
from axon.sentence_window import (
    SentenceRecord,
    SentenceVectorStore,
    SentenceWindowIndex,
    is_eligible,
    segment_chunk,
    segment_text,
)


def test_is_eligible():
    # Eligible: prose chunk
    assert is_eligible({"metadata": {"source_class": "prose"}}) is True
    # Ineligible: code
    assert is_eligible({"metadata": {"source_class": "code"}}) is False
    # Ineligible: raptor summary
    assert is_eligible({"metadata": {"chunk_kind": "raptor"}}) is False
    # Ineligible: raptor level
    assert is_eligible({"metadata": {"raptor_level": 1}}) is False
    # Ineligible: parent marker
    assert is_eligible({"metadata": {"chunk_kind": "parent"}}) is False


def test_segment_text():
    # Temporarily reduce min sentence chars for testing
    orig_min = sw._MIN_SENTENCE_CHARS
    sw._MIN_SENTENCE_CHARS = 3
    try:
        text = "Sentence one. Sentence two! Sentence three? Stub. Final."
        segments = segment_text(text)
        # "Stub." is 5 chars >= 3. "Final." is 6 chars >= 3.
        assert len(segments) == 5
        assert segments[0] == "Sentence one."
        assert segments[1] == "Sentence two!"
        assert segments[2] == "Sentence three?"
        assert segments[3] == "Stub."
        assert segments[4] == "Final."
    finally:
        sw._MIN_SENTENCE_CHARS = orig_min


def test_segment_chunk():
    orig_min = sw._MIN_SENTENCE_CHARS
    sw._MIN_SENTENCE_CHARS = 3
    try:
        chunk = {
            "id": "chunk1",
            "text": "First sentence. Second sentence. Third sentence.",
            "metadata": {"source": "test.txt", "source_class": "prose"},
        }
        records = segment_chunk(chunk)
        assert len(records) == 3
        assert records[0].sentence_id == "chunk1_s0"
        assert records[0].chunk_id == "chunk1"
        assert records[0].sentence_idx == 0
        assert records[0].total_sentences == 3
        assert records[0].text == "First sentence."
    finally:
        sw._MIN_SENTENCE_CHARS = orig_min


def test_sentence_window_index_reconstruction():
    orig_min = sw._MIN_SENTENCE_CHARS
    sw._MIN_SENTENCE_CHARS = 2
    try:
        index = SentenceWindowIndex()
        chunk = {"id": "c1", "text": "S0. S1. S2. S3. S4. S5.", "metadata": {"source": "s.txt"}}
        records = segment_chunk(chunk)
        index.add_records(records)

        # Window size 1 around S2 -> S1 S2 S3
        window = index.get_window("c1_s2", window_size=1)
        assert window == "S1. S2. S3."

        # Window size 2 around S0 -> S0 S1 S2
        window = index.get_window("c1_s0", window_size=2)
        assert window == "S0. S1. S2."

        # Window size 5 around S5 -> S0 S1 S2 S3 S4 S5
        window = index.get_window("c1_s5", window_size=5)
        assert window == "S0. S1. S2. S3. S4. S5."
    finally:
        sw._MIN_SENTENCE_CHARS = orig_min


def test_sentence_window_index_persistence(tmp_path):
    index = SentenceWindowIndex()
    records = [SentenceRecord("c1_s0", "c1", "s.txt", 0, 1, "Hello.")]
    index.add_records(records)
    index.save(tmp_path)

    new_index = SentenceWindowIndex()
    new_index.load(tmp_path)
    assert len(new_index) == 1
    assert new_index.get_record("c1_s0").text == "Hello."


def test_sentence_vector_store(tmp_path):
    vs = SentenceVectorStore(tmp_path)
    ids = ["s1", "s2"]
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    meta = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]

    vs.add(ids, vecs, meta)
    assert len(vs) == 2

    # Search for something close to [1.0, 0.1]
    results = vs.search([1.0, 0.1], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "s1"
    assert results[0]["score"] > 0.9

    vs.save()

    new_vs = SentenceVectorStore(tmp_path)
    new_vs.load()
    assert len(new_vs) == 2
    assert new_vs._ids == ["s1", "s2"]


def test_citation_integrity():
    # Story 1.5 requirement: regression checks for citation integrity
    # Ensure that sentence records always point back to the stable chunk_id
    chunk = {
        "id": "stable_chunk_id",
        "text": "Sent 1. Sent 2.",
        "metadata": {"source": "important.txt"},
    }
    records = segment_chunk(chunk)
    for r in records:
        assert r.chunk_id == "stable_chunk_id"
        assert r.source == "important.txt"
