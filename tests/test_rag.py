from pathlib import Path
from src.local_rag.bm25_index import BM25Index
from src.local_rag.faiss_builder import FAISSIndex
from src.hybrid_retrieval.hybrid_retriever_with_DAT import DATAlphaCalculator, HybridRetrieverWithDAT
from src.pipeline.science_rag_pipeline import ScienceRAGPipeline


def test_bm25_loads():
    bm25 = BM25Index.load(
        Path("data/chunks/local/bm25_index.json"),
        Path("data/chunks/local/bm25_meta.json")
    )
    assert len(bm25.meta.doc_ids) > 100
    res = bm25.search("transformer model", top_k=3)
    assert len(res) == 3


def test_dense_loads():
    dense = FAISSIndex.load(
        Path("data/faiss/local_dense.index"),
        Path("data/faiss/local_dense_meta.json"),
    )
    vec = [0.1] * dense.dim
    res = dense.search(vec, top_k=3)
    assert len(res) == 3


def test_dat_alpha():
    dat = DATAlphaCalculator()

    alpha = dat.compute_alpha(
        "test query",
        bm25_top1_text="irrelevant text",
        dense_top1_text="relevant text about transformers"
    )
    assert 0 <= alpha <= 1


def test_hybrid_retriever():
    pipeline = ScienceRAGPipeline()
    res = pipeline.answer_local_hybrid("quantization in transformers")
    assert len(res.contexts) > 0
    assert res.answer


def test_full_rag_local():
    pipeline = ScienceRAGPipeline()
    res = pipeline.answer_local("what is attention?")
    assert res.answer
    assert len(res.contexts) > 0
