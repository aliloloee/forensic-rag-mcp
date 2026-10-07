"""Email-level evaluation of a results JSON against the annotated ground truth.

Ported from compute_retrieval_metrics / classification_metrics in
Thesis codes/7- Inference/script.ipynb in the thesis repo
(https://github.com/aliloloee/forensic-analysis).

Usage:
  python -m forensic_rag.evaluate results/H3_20261004_120000.json [more.json ...]
"""

import argparse
import glob
import json
from pathlib import Path

from forensic_rag.hypotheses import get_hypothesis


def prf(predicted: set, truth: set) -> dict:
    tp = len(predicted & truth)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(truth) if truth else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "tp": tp,
            "predicted": len(predicted), "truth": len(truth)}


def evaluate(report: dict) -> dict:
    h = get_hypothesis(report["hypothesis_id"])
    high_gt, medium_gt = set(h["high"]), set(h["medium"])
    full_gt = high_gt | medium_gt

    retrieved = {int(e["email_id"]) for e in report["emails"]}
    pred_high = {int(e["email_id"]) for e in report["emails"] if e["strength"] == "high"}
    pred_medium = {int(e["email_id"]) for e in report["emails"] if e["strength"] == "medium"}
    useful = pred_high | pred_medium

    return {
        "retrieval": prf(retrieved, full_gt),
        "inference_high": prf(pred_high, high_gt),
        "inference_medium": prf(pred_medium, medium_gt),
        "inference_useful": prf(useful, full_gt),
        "missed_by_retrieval": sorted(full_gt - retrieved),
        "relevant_labelled_low": sorted((full_gt & retrieved) - useful),
        "false_useful": sorted(useful - full_gt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="result files or glob patterns, e.g. results/*.json")
    args = parser.parse_args()

    # PowerShell does not expand wildcards for native commands, so expand them here
    paths = []
    for pattern in args.results:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"no files match {pattern}")
        paths.extend(Path(m) for m in matches)

    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("hypothesis_id"):
            print(f"\n=== {path.name}: skipped, custom hypothesis has no ground truth ===")
            continue
        metrics = evaluate(report)
        print(f"\n=== {path.name} ({report['hypothesis_id']}, dataset {report.get('dataset', '?')}) ===")
        for name in ("retrieval", "inference_high", "inference_medium", "inference_useful"):
            m = metrics[name]
            print(f"{name:18s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                  f"(tp={m['tp']}, pred={m['predicted']}, gt={m['truth']})")
        for name in ("missed_by_retrieval", "relevant_labelled_low", "false_useful"):
            print(f"{name:22s} {metrics[name]}")


if __name__ == "__main__":
    main()
