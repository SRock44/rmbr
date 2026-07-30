"""Semantic retrieval quality: recall@k on real, meaningful text.

This is deliberately different from bench/run.py, which uses synthetic
random vectors to isolate storage/ANN engine performance — that
methodology is explicitly incapable of answering "which embedding model
actually understands text better," because random vectors carry no
semantic content to understand. This script measures exactly that:
given a query and a handful of topically-similar passages, does the
embedder rank the actually-correct passage above the near-miss
distractors?

**Honestly scoped:** 150 hand-written examples (bench/quality_data.py),
50 each across memory/docs/code, 10 sub-themes of 5 per category for
real topical diversity — not a large-scale published IR benchmark
(BEIR, MTEB, etc.), but no longer a handful of anecdotes either. Sized
to support a real default-model decision on rmbr's actual use case, not
to be a definitive academic result. The data is plain Python, checked
in and readable, not hidden in a black-box file.

    python bench/quality.py                       # benches the local default
    python bench/quality.py --models candidates    # compares a shortlist of local models
    OPENAI_API_KEY=sk-... python bench/quality.py --with-openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from quality_data import EXAMPLES  # noqa: E402
from rmbr.embed import Embedder, FastEmbedEmbedder  # noqa: E402


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def evaluate(embedder: Embedder, examples: list[dict], k: int = 1) -> dict:
    """Returns {"recall_at_k": float, "by_category": {...}, "misses": [...]}."""
    hits = 0
    by_category: dict[str, list[int]] = {}
    misses = []

    for ex in examples:
        passages = [ex["correct"]] + ex["distractors"]
        query_vector = embedder.embed([ex["query"]])[0]
        passage_vectors = embedder.embed(passages)
        scores = [_cosine(query_vector, v) for v in passage_vectors]
        ranked = sorted(range(len(passages)), key=lambda i: -scores[i])
        correct_rank = ranked.index(0)  # index 0 is always the correct passage
        hit = correct_rank < k
        hits += hit
        by_category.setdefault(ex["category"], []).append(hit)
        if not hit:
            misses.append({"query": ex["query"], "correct": ex["correct"], "rank": correct_rank})

    return {
        "recall_at_k": hits / len(examples),
        "by_category": {cat: sum(v) / len(v) for cat, v in by_category.items()},
        "misses": misses,
    }


def _print_report(name: str, result: dict, verbose: bool = False) -> None:
    print(f"\n{name}")
    print(f"  overall recall@1: {result['recall_at_k']:.3f} ({len(EXAMPLES)} examples)")
    for cat, score in result["by_category"].items():
        n = sum(1 for ex in EXAMPLES if ex["category"] == cat)
        print(f"    {cat:<10} recall@1: {score:.3f}  ({n} examples)")
    if verbose and result["misses"]:
        print("  missed:")
        for miss in result["misses"]:
            print(f"    - {miss['query']!r} -> correct passage ranked #{miss['rank'] + 1}, not #1")


# Same size class as the current default (bge-small-en-v1.5, 384-dim,
# 67MB) plus a couple of reference points either side - every one of
# these is a local ONNX model fastembed already knows how to fetch and
# run, so comparing them costs nothing but the one-time download and a
# few seconds of local inference. No API key anywhere in this list.
CANDIDATE_MODELS = [
    "BAAI/bge-small-en-v1.5",  # current default
    "snowflake/snowflake-arctic-embed-xs",  # smaller footprint, retrieval-focused training
    "snowflake/snowflake-arctic-embed-s",  # same dim as bge-small, retrieval-focused training
    "sentence-transformers/all-MiniLM-L6-v2",  # older, widely-used baseline for reference
    "jinaai/jina-embeddings-v2-small-en",  # 8192-token context vs bge's 512 - relevant if chunks run long
    "BAAI/bge-base-en-v1.5",  # 3x the size - reference point for "how much does bigger buy you"
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        default="BAAI/bge-small-en-v1.5",
        help=f"comma-separated fastembed model names to compare, or 'candidates' for a preset shortlist "
        f"({', '.join(CANDIDATE_MODELS)})",
    )
    parser.add_argument(
        "--with-openai", action="store_true", help="also evaluate OpenAIEmbedder (needs OPENAI_API_KEY)"
    )
    parser.add_argument("--verbose", action="store_true", help="list every missed example, not just the summary")
    args = parser.parse_args()

    model_names = CANDIDATE_MODELS if args.models == "candidates" else args.models.split(",")

    print(f"Evaluating on {len(EXAMPLES)} hand-written (query, correct, distractors) examples...")

    results = []
    for model_name in model_names:
        embedder = FastEmbedEmbedder(model_name=model_name.strip())
        result = evaluate(embedder, EXAMPLES)
        _print_report(f"local: {embedder.model_name}", result, verbose=args.verbose)
        results.append((embedder.model_name, result))

    if args.with_openai:
        from rmbr.embed import OpenAIEmbedder

        openai_embedder = OpenAIEmbedder()
        openai_result = evaluate(openai_embedder, EXAMPLES)
        _print_report(f"OpenAIEmbedder ({openai_embedder.model_name})", openai_result, verbose=args.verbose)
        results.append((f"OpenAIEmbedder ({openai_embedder.model_name})", openai_result))

    if len(results) > 1:
        print(f"\n{'model':<45}{'overall':>10}{'memory':>10}{'docs':>10}{'code':>10}")
        for name, result in results:
            by_cat = result["by_category"]
            print(
                f"{name:<45}{result['recall_at_k']:>10.3f}"
                f"{by_cat.get('memory', float('nan')):>10.3f}"
                f"{by_cat.get('docs', float('nan')):>10.3f}"
                f"{by_cat.get('code', float('nan')):>10.3f}"
            )


if __name__ == "__main__":
    main()
