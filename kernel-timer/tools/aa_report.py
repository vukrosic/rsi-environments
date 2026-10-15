"""Summarize pair3 A/A receipts (kernel-koth-paired-v1).
  python tools/aa_report.py results/aa-v4/*.json [--gate 0.02] [--json]
"""
import argparse, glob, json, math, statistics
from pathlib import Path


def median(values):
    return statistics.median(values) if values else None


def phase_report(phase):
    pairs = phase.get("pairs", [])
    ratios = [p["candidate_ms"] / p["incumbent_ms"] for p in pairs]
    clocks = [p["clock_sm_mhz"] for p in pairs if p.get("clock_sm_mhz") is not None]
    middle = median(clocks)
    low = [p for p in pairs if middle is not None and p.get("clock_sm_mhz") is not None
           and p["clock_sm_mhz"] < 0.9 * middle]
    kept = [p for p in pairs if p not in low]
    return {
        "n": len(pairs),
        "candidate_incumbent_median": median(ratios),
        "order_medians": {first + "-first": median([
            ratio for p, ratio in zip(pairs, ratios) if p["order"][0] == first
        ]) for first in ("incumbent", "candidate")},
        "anchor_incumbent_median": median([p["anchor_ms"] / p["incumbent_ms"] for p in pairs]),
        "anchor_candidate_median": median([p["anchor_ms"] / p["candidate_ms"] for p in pairs]),
        "clock_sm_mhz": {"min": min(clocks) if clocks else None, "median": middle,
                         "max": max(clocks) if clocks else None},
        "clock_read": sorted({p.get("clock_read") for p in pairs}, key=str),
        "low_clock_count": len(low),
        "incumbent_mean_ms": {name: statistics.mean([p["incumbent_ms"] for p in group])
                              if group else None for name, group in (("low", low), ("kept", kept))},
    }


def summarize(path, gate=0.02):
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != "kernel-koth-paired-v1":
        raise ValueError("expected schema kernel-koth-paired-v1")
    report = {"file": str(path), "status": data.get("status")}
    if report["status"] != "ok":
        report["failure"] = data.get("failure") or {}
        return report
    for name in ("interaction", "validation"):
        report[name] = phase_report(data.get(name) or {})
    ratio = report["validation"]["candidate_incumbent_median"]
    report["gate"] = gate
    report["copy_risk"] = ratio is not None and (ratio <= 1 - gate or ratio >= 1 + gate)
    report["gate_distance"] = min(abs(ratio - (1 - gate)), abs(ratio - (1 + gate))) if ratio is not None else None
    return report


def number(value):
    return "n/a" if value is None else f"{value:.6g}"


def format_report(report):
    title = f"{report['file']}: status={report['status']}"
    if report["status"] != "ok":
        failure = report["failure"]
        return title + " " + " ".join(f"{key}={failure.get(key, 'n/a')}"
                                      for key in ("blame", "stage", "reason"))
    if report["copy_risk"]:
        title += " COPY-RISK"
    distance = report["gate_distance"]
    title += f"; {distance:.2%} from the gate" if distance is not None else "; gate distance=n/a"
    lines = [title]
    for name in ("interaction", "validation"):
        phase = report[name]
        order, clock, means = phase["order_medians"], phase["clock_sm_mhz"], phase["incumbent_mean_ms"]
        lines.append(f"  {name}: n={phase['n']} candidate/incumbent={number(phase['candidate_incumbent_median'])}"
                     f" (incumbent-first={number(order['incumbent-first'])}, candidate-first={number(order['candidate-first'])})"
                     f" anchor/incumbent={number(phase['anchor_incumbent_median'])}"
                     f" anchor/candidate={number(phase['anchor_candidate_median'])}")
        lines.append("    clock MHz min/median/max=" + "/".join(number(clock[k]) for k in ("min", "median", "max"))
                     + f" clock_read={json.dumps(phase['clock_read'])} low-clock={phase['low_clock_count']}"
                     + f" incumbent mean ms low/kept={number(means['low'])}/{number(means['kept'])}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="receipt files or globs")
    parser.add_argument("--json", action="store_true", help="emit a JSON list")
    parser.add_argument("--gate", type=float, default=0.02, help="A/A gate around 1 (default: 0.02)")
    args = parser.parse_args()
    if not math.isfinite(args.gate) or not 0 <= args.gate < 1:
        parser.error("--gate must be finite and in [0, 1)")
    paths = []
    for pattern in args.paths:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"no files match {pattern!r}")
        paths.extend(matches)
    try:
        reports = [summarize(path, args.gate) for path in dict.fromkeys(paths)]
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        parser.error(str(error))
    print(json.dumps(reports, indent=2, allow_nan=False) if args.json else
          "\n\n".join(format_report(report) for report in reports))


if __name__ == "__main__":
    main()
