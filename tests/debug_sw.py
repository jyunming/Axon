from axon.sentence_window import SentenceWindowIndex, segment_chunk


def test_sentence_window_index_reconstruction_debug():
    index = SentenceWindowIndex()
    chunk = {"id": "c1", "text": "S0. S1. S2. S3. S4. S5.", "metadata": {"source": "s.txt"}}
    records = segment_chunk(chunk)
    print(f"DEBUG: records={records}")
    index.add_records(records)
    print(f"DEBUG: index._records keys={index._records.keys()}")

    # Window size 1 around S2 -> S1 S2 S3
    window = index.get_window("c1_s2", window_size=1)
    print(f"DEBUG: window for c1_s2={window}")
    assert window == "S1. S2. S3."
