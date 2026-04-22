from axon.config import AxonConfig
from axon.embeddings import OpenEmbedding


def test_dry_run_embedding_skip(monkeypatch):
    """When AXON_DRY_RUN is set, OpenEmbedding must not load external models and should
    return zero vectors of the configured dimension instead of calling provider SDKs.
    """
    monkeypatch.setenv("AXON_DRY_RUN", "1")
    cfg = AxonConfig()
    cfg.embedding_provider = "ollama"
    cfg.embedding_model = "all-MiniLM-L6-v2"
    emb = OpenEmbedding(cfg)
    out = emb.embed(["hello", "world"])
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(len(vec) == emb.dimension for vec in out)
    assert all(all(x == 0.0 for x in vec) for vec in out)
