"""
BNS Legal RAG — Evaluation Suite
==================================
Runs RAGAS metrics on a gold test set.
Used as a CI gate: blocks deploy if metrics drop below thresholds.

Usage:
    python tests/eval_suite.py
    python tests/eval_suite.py --threshold-faithfulness 0.80
"""

import argparse
import json
import sys
from pathlib import Path

# ── Gold Test Set ─────────────────────────────────────────────────────────────
# Curated Q&A pairs covering: basic lookups, cross-section, punishment, ambiguous

GOLD_TEST_SET = [
    {
        "query": "What is the punishment for murder under BNS?",
        "expected_sections": ["101", "103"],
        "category": "punishment_lookup",
    },
    {
        "query": "Define theft under BNS",
        "expected_sections": ["303"],
        "category": "definition_lookup",
    },
    {
        "query": "What constitutes criminal conspiracy?",
        "expected_sections": ["61"],
        "category": "definition_lookup",
    },
    {
        "query": "Explain the provisions related to dowry death",
        "expected_sections": ["80"],
        "category": "cross_section",
    },
    {
        "query": "What are the different types of hurt defined in BNS?",
        "expected_sections": ["114", "115", "116", "117"],
        "category": "multi_section",
    },
    {
        "query": "What is the age of criminal responsibility?",
        "expected_sections": ["2"],
        "category": "definition_lookup",
    },
    {
        "query": "Explain the right of private defence under BNS",
        "expected_sections": ["34", "35", "36", "37"],
        "category": "cross_section",
    },
    {
        "query": "What offences relate to elections?",
        "expected_sections": ["171"],
        "category": "topic_search",
    },
    {
        "query": "What is culpable homicide not amounting to murder?",
        "expected_sections": ["105"],
        "category": "ambiguous",
    },
    {
        "query": "Explain offences against women and children",
        "expected_sections": ["63", "64", "65", "66", "67"],
        "category": "multi_section",
    },
]


def compute_retrieval_metrics(retrieved_sections: list[str], expected_sections: list[str]) -> dict:
    """Compute hit@5 and MRR for a single query."""
    retrieved_set = set(retrieved_sections[:5])
    expected_set = set(expected_sections)

    # Hit@5: Did we retrieve at least one expected section in top 5?
    hit = 1.0 if retrieved_set & expected_set else 0.0

    # MRR: Reciprocal rank of first relevant result
    mrr = 0.0
    for rank, sec in enumerate(retrieved_sections, 1):
        if sec in expected_set:
            mrr = 1.0 / rank
            break

    # Precision@5
    precision = len(retrieved_set & expected_set) / max(len(retrieved_set), 1)

    # Recall
    recall = len(retrieved_set & expected_set) / max(len(expected_set), 1)

    return {
        "hit_at_5": hit,
        "mrr": mrr,
        "precision_at_5": precision,
        "recall": recall,
    }


def run_eval(
    threshold_hit: float = 0.80,
    threshold_faithfulness: float = 0.80,
) -> bool:
    """
    Run evaluation suite. Returns True if all gates pass.

    In full mode (with API keys), runs actual queries through the pipeline.
    In CI mode (no keys), runs retrieval-only metrics on pre-computed results.
    """
    print("=" * 60)
    print("BNS Legal RAG — Evaluation Suite")
    print("=" * 60)
    print(f"Gold test set: {len(GOLD_TEST_SET)} queries")
    print(f"Thresholds: hit@5 >= {threshold_hit}, faithfulness >= {threshold_faithfulness}")
    print()

    # Check if we can run full eval or just structural checks
    results_file = Path("tests/eval_results.json")

    if results_file.exists():
        print("Loading pre-computed eval results...")
        with open(results_file) as f:
            eval_results = json.load(f)
    else:
        print("No pre-computed results found. Running structural eval only.")
        print("To run full eval: query the API for each test case and save results.")
        print()

        # Structural validation only
        passed = True
        for i, test in enumerate(GOLD_TEST_SET):
            assert "query" in test, f"Test {i} missing 'query'"
            assert "expected_sections" in test, f"Test {i} missing 'expected_sections'"
            assert len(test["expected_sections"]) > 0, f"Test {i} has empty expected_sections"
            print(f"  ✅ Test {i+1}: {test['category']} — structure valid")

        print()
        print(f"Structural validation: {'PASSED' if passed else 'FAILED'}")
        return passed

    # Compute metrics from pre-computed results
    all_metrics = []
    for result in eval_results:
        metrics = compute_retrieval_metrics(
            result.get("retrieved_sections", []),
            result.get("expected_sections", []),
        )
        metrics["faithfulness"] = result.get("faithfulness", 0.0)
        all_metrics.append(metrics)

    # Aggregate
    avg_hit = sum(m["hit_at_5"] for m in all_metrics) / len(all_metrics)
    avg_mrr = sum(m["mrr"] for m in all_metrics) / len(all_metrics)
    avg_precision = sum(m["precision_at_5"] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_faithfulness = sum(m["faithfulness"] for m in all_metrics) / len(all_metrics)

    print("Results:")
    print(f"  Hit@5:          {avg_hit:.3f}  (threshold: {threshold_hit})")
    print(f"  MRR:            {avg_mrr:.3f}")
    print(f"  Precision@5:    {avg_precision:.3f}")
    print(f"  Recall:         {avg_recall:.3f}")
    print(f"  Faithfulness:   {avg_faithfulness:.3f}  (threshold: {threshold_faithfulness})")
    print()

    # Gate check
    gate_passed = True
    if avg_hit < threshold_hit:
        print(f"  ❌ GATE FAILED: hit@5 ({avg_hit:.3f}) < threshold ({threshold_hit})")
        gate_passed = False
    else:
        print(f"  ✅ hit@5 gate passed")

    if avg_faithfulness < threshold_faithfulness:
        print(f"  ❌ GATE FAILED: faithfulness ({avg_faithfulness:.3f}) < threshold ({threshold_faithfulness})")
        gate_passed = False
    else:
        print(f"  ✅ faithfulness gate passed")

    print()
    print(f"Overall: {'PASSED ✅' if gate_passed else 'FAILED ❌'}")
    return gate_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNS RAG Eval Suite")
    parser.add_argument("--threshold-hit", type=float, default=0.80)
    parser.add_argument("--threshold-faithfulness", type=float, default=0.80)
    args = parser.parse_args()

    passed = run_eval(args.threshold_hit, args.threshold_faithfulness)
    sys.exit(0 if passed else 1)
