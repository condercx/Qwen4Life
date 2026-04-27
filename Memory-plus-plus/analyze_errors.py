#!/usr/bin/env python3
"""Analyze v8 benchmark errors and categorize failure modes."""
import re
import sys
from collections import Counter, defaultdict

def categorize_error(pred: str, gt: str, qtype: str) -> str:
    """Categorize a zero-F1 error into failure modes."""
    pred_l = pred.lower().strip()
    gt_l = gt.lower().strip()

    # IDK when answer exists
    if "don't know" in pred_l or "not mentioned" in pred_l or "no information" in pred_l:
        if "don't know" in gt_l or "not mention" in gt_l:
            return "correct_idk"  # both IDK
        return "false_idk"  # model says IDK but answer exists

    # Model answered but GT is IDK
    if "don't know" in gt_l or "not mention" in gt_l:
        return "missed_idk"  # model answered but should have said IDK

    # Both have content but don't match
    # Check if it's a counting error (both are numbers)
    pred_nums = re.findall(r'\d+', pred_l)
    gt_nums = re.findall(r'\d+', gt_l)
    if pred_nums and gt_nums and qtype in ("multi-session", "temporal-reasoning"):
        return "wrong_number"

    # Wrong entity/fact
    return "wrong_answer"

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_kg_v8_run.log"

    errors = defaultdict(list)
    type_errors = defaultdict(lambda: Counter())

    with open(log_file) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match: [  3/500] ✗ [single-session-user] F1=0.00
        m = re.search(r'\[(\w[\w-]*)\]\s+F1=([\d.]+)', line)
        if m and '✗' in line:
            qtype = m.group(1)
            f1 = float(m.group(2))
            # Next line has prediction and ground truth
            if i + 1 < len(lines):
                ans_line = lines[i + 1].strip()
                am = re.match(r'答:\s*(.*?)\s*\|\s*真:\s*(.*)', ans_line)
                if am:
                    pred = am.group(1).strip()
                    gt = am.group(2).strip()
                    cat = categorize_error(pred, gt, qtype)
                    errors[qtype].append((cat, pred[:50], gt[:50]))
                    type_errors[qtype][cat] += 1
        i += 1

    print("=" * 70)
    print("Error Analysis Report")
    print("=" * 70)

    total_errors = sum(len(v) for v in errors.values())
    print(f"\nTotal zero-F1 errors: {total_errors}")

    # Overall error distribution
    all_cats = Counter()
    for qtype, cats in type_errors.items():
        for cat, count in cats.items():
            all_cats[cat] += count

    print("\nOverall error categories:")
    for cat, count in all_cats.most_common():
        pct = count / total_errors * 100
        print(f"  {cat:20s}: {count:3d} ({pct:.1f}%)")

    # Per-type breakdown
    print("\nPer-type error breakdown:")
    for qtype in sorted(type_errors.keys()):
        cats = type_errors[qtype]
        total = sum(cats.values())
        print(f"\n  {qtype} ({total} errors):")
        for cat, count in cats.most_common():
            print(f"    {cat:20s}: {count:3d}")

    # Sample errors for each category
    print("\n" + "=" * 70)
    print("Sample errors by category")
    print("=" * 70)
    for cat in all_cats.keys():
        print(f"\n--- {cat} ---")
        shown = 0
        for qtype, errs in errors.items():
            for c, pred, gt in errs:
                if c == cat and shown < 3:
                    print(f"  [{qtype}] pred={pred!r}  gt={gt!r}")
                    shown += 1
            if shown >= 3:
                break

if __name__ == "__main__":
    main()
