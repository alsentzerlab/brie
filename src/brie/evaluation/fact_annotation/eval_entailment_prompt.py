#!/usr/bin/env python
"""
eval_entailment_prompt.py

Score the CURRENT entailment prompt against your own labels, on a small
hand-annotated sample. Built for a manual prompt-tuning loop:

    edit RECALL_PROMPT / PRECISION_PROMPT in score_facts_batch.py
      -> python eval_entailment_prompt.py --annotations fact_annotations_*.json
      -> read the stats, edit again

The prompts are imported live from score_facts_batch.py, so whatever you just
typed there is what gets tested — no copy-paste step. (--prompts overrides with a
YAML file if you'd rather not touch the source between runs.)

Facts are NOT re-extracted. The atomic fact lists come from the annotation export
exactly as the annotator saw them, so the judge scores the same fixed lists you
labelled and index i always means the same fact.

Inference is the interactive/online Vertex API (utils.send_single_message), one call
per record per direction, run concurrently — seconds per iteration, not a batch job.

INPUT: the JSON exported by fact_annotation.html ("Export JSON"). Each record has
  {record_id, question_id, source_name, model,
   recall_items:    [{idx, fact, entailed, note}],   <- REFERENCE facts
   precision_items: [{idx, fact, entailed, note}]}   <- CANDIDATE facts
Merged exports (merge_fact_annotations.py) work too; extra fields are ignored.
Facts left unmarked (entailed = null) are excluded from every statistic.

Usage:
  python -m brie.evaluation.fact_annotation.eval_entailment_prompt --annotations ANNOTATIONS.json
  python eval_entailment_prompt.py --annotations ann/ --repeat 3 --out run1.csv
"""

import argparse
import asyncio
import base64
import csv
import glob
import json
import re
import sys
from pathlib import Path

from ..score_facts_batch import (
    PRECISION_PROMPT,
    RECALL_PROMPT,
    SYSTEM_PROMPT,
    _parse_entailment_response,
)
from ..utils import send_single_message

# "r" = recall direction: which REFERENCE facts are entailed by the candidate answer.
# "p" = precision direction: which CANDIDATE facts are supported by the reference.
DIRECTIONS = ("r", "p")
DIR_LABEL  = {"r": "recall", "p": "precision"}

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")


# ── Load ──────────────────────────────────────────────────────────────────────
def _expand(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            out.extend(sorted(p.glob("*.json")))
        else:
            matched = [Path(m) for m in glob.glob(pat)]
            out.extend(sorted(matched) if matched else [p])
    return out


def load_records(patterns: list[str]) -> list[dict]:
    """Load annotation exports into {record_id, facts, human} records.

    facts[d]  = list of fact strings, positionally indexed by the annotation `idx`
    human[d]  = list of True/False/None marks aligned to facts[d]
    """
    by_id: dict[str, dict] = {}
    for path in _expand(patterns):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARNING: cannot read {path}: {e}")
            continue
        if not isinstance(data, list):
            print(f"  WARNING: {path} is not a JSON list — skipping")
            continue
        for a in data:
            if not isinstance(a, dict) or not a.get("record_id"):
                continue
            rec = {"record_id": a["record_id"], "question_id": a.get("question_id", ""),
                   "source_name": a.get("source_name", ""), "model": a.get("model", ""),
                   "facts": {}, "human": {}, "note": {}}
            for d, key in (("r", "recall_items"), ("p", "precision_items")):
                items = sorted((a.get(key) or []), key=lambda x: x.get("idx", 0))
                rec["facts"][d] = [str(it.get("fact", "")) for it in items]
                rec["human"][d] = [None if it.get("entailed") is None else bool(it["entailed"])
                                   for it in items]
                rec["note"][d]  = [str(it.get("note") or "") for it in items]
            if not rec["facts"]["r"] and not rec["facts"]["p"]:
                continue
            by_id[rec["record_id"]] = rec          # last file wins on duplicate ids
        print(f"  {path}: {len(data)} annotations")
    # Insertion order = export order, which is the UI card order for completed records.
    # load_card_order() overrides this with the authoritative order when available.
    return list(by_id.values())


def load_card_order(records_arg: str | None, map_arg: str | None) -> dict[str, int]:
    """record_id -> 0-based card position in the annotation UI.

    The UI numbers cards by their position in PRELOADED_DATA and shows i+1 in the
    sidebar. Prefer decoding that array straight out of the generated HTML; the
    map.json sidecar carries the same order. Returns {} if neither was given, in
    which case the caller falls back to export order.
    """
    if records_arg:
        html = Path(records_arg).read_text(encoding="utf-8")
        m = re.search(r'const PRELOADED_DATA = JSON\.parse\(atob\("([^"]+)"\)\);', html)
        if not m:
            sys.exit(f"ERROR: no PRELOADED_DATA found in {records_arg} — is it the generated HTML?")
        data = json.loads(base64.b64decode(m.group(1)).decode("utf-8"))
        return {r["record_id"]: i for i, r in enumerate(data) if r.get("record_id")}
    if map_arg:
        mapping = json.loads(Path(map_arg).read_text(encoding="utf-8"))
        return {rid: i for i, rid in enumerate(mapping)}
    return {}


def load_prompts(path: str | None) -> dict[str, str]:
    """Live prompts from score_facts_batch.py, or a YAML override with recall/precision/system."""
    prompts = {"system": SYSTEM_PROMPT, "recall": RECALL_PROMPT, "precision": PRECISION_PROMPT}
    if path:
        import yaml
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k in prompts:
            if cfg.get(k):
                prompts[k] = cfg[k]
    for key in ("recall", "precision"):
        for field in ("{REFERENCE_FACTS}", "{CANDIDATE_FACTS}"):
            if field not in prompts[key]:
                sys.exit(f"ERROR: the {key} prompt is missing the {field} placeholder.")
    return prompts


# ── Inference ─────────────────────────────────────────────────────────────────
def _numbered(facts: list[str]) -> str:
    return "\n".join(f"{i}. {f}" for i, f in enumerate(facts))


async def _judge(rec: dict, d: str, prompts: dict[str, str], model: str,
                 sem: asyncio.Semaphore) -> tuple[str, str, set[int] | None, str]:
    template = prompts["recall"] if d == "r" else prompts["precision"]
    user = template.format(REFERENCE_FACTS=_numbered(rec["facts"]["r"]),
                           CANDIDATE_FACTS=_numbered(rec["facts"]["p"]))
    async with sem:
        try:
            raw = await send_single_message(user, prompts["system"], model_id=model)
        except Exception as e:                    # one bad call shouldn't kill the run
            return rec["record_id"], d, None, f"<error: {e}>"
    parsed = _parse_entailment_response(raw)
    return rec["record_id"], d, (None if parsed is None else set(parsed)), raw


async def _one_pass(records: list[dict], prompts: dict[str, str], model: str,
                    concurrency: int) -> dict[tuple[str, str], tuple[set[int] | None, str]]:
    sem   = asyncio.Semaphore(concurrency)
    tasks = [_judge(r, d, prompts, model, sem) for r in records for d in DIRECTIONS]
    return {(rid, d): (idxs, raw) for rid, d, idxs, raw in await asyncio.gather(*tasks)}


# ── Stats ─────────────────────────────────────────────────────────────────────
def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "  n/a"
    return f"{100*x:5.1f}%" if pct else f"{x:5.3f}"


def _cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Judge vs human on the binary Present/Absent call. fp = judge Present, human Absent."""
    n = tp + fp + fn + tn
    if not n:
        return None
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compute(records: list[dict], judged: dict) -> dict:
    """Per-fact comparison + all aggregate statistics."""
    rows: list[dict] = []          # one per graded fact
    per_record: list[dict] = []
    errors: list[str] = []

    for rec in records:
        rec_row = {"card": rec["card"], "record_id": rec["record_id"],
                   "question_id": rec["question_id"],
                   "source_name": rec["source_name"], "exact_match": True,
                   "n_graded": 0, "n_agree": 0, "human": {}, "judge": {},
                   "dir_agree": {}}   # d -> (n_agree, n_graded) for macro averaging
        for d in DIRECTIONS:
            facts = rec["facts"][d]
            marks = rec["human"][d]
            idxs, raw = judged.get((rec["record_id"], d), (None, ""))
            if idxs is None:
                errors.append(f"{rec['record_id']}/{DIR_LABEL[d]}: unparseable response — {raw[:120]}")
                rec_row["exact_match"] = False
            # Both raters are measured on the SAME facts — those you labelled — so the
            # two columns are directly comparable.
            h_num = h_den = j_num = j_den = d_agree = d_graded = 0
            for i, fact in enumerate(facts):
                human = marks[i] if i < len(marks) else None
                judge = None if idxs is None else (i in idxs)
                if human is None:
                    continue                       # unmarked: excluded everywhere
                h_den += 1
                h_num += human
                if judge is None:
                    continue
                j_den += 1
                j_num += judge
                agree = (human == judge)
                rec_row["n_graded"] += 1
                rec_row["n_agree"]  += agree
                d_graded += 1
                d_agree  += agree
                if not agree:
                    rec_row["exact_match"] = False
                rows.append({"card": rec["card"], "card_index": rec["card_index"],
                             "record_id": rec["record_id"], "question_id": rec["question_id"],
                             "source_name": rec["source_name"], "direction": DIR_LABEL[d],
                             "idx": i, "fact": fact, "human": human, "judge": judge,
                             "agree": agree,
                             "your_note": rec["note"][d][i] if i < len(rec["note"][d]) else ""})
            rec_row["human"][d] = _rate(h_num, h_den)
            rec_row["judge"][d] = _rate(j_num, j_den)
            rec_row["dir_agree"][d] = (d_agree, d_graded)
        per_record.append(rec_row)

    # (A) The metric itself, as each rater would report it.
    #     micro = over all facts; macro = mean of per-record rates (how score_facts_batch does it)
    scores: dict[str, dict[str, float | None]] = {}
    for who in ("human", "judge"):
        for d in DIRECTIONS:
            num = sum(1 for r in rows if r["direction"] == DIR_LABEL[d] and r[who])
            den = sum(1 for r in rows if r["direction"] == DIR_LABEL[d])
            vals = [rec[who][d] for rec in per_record if rec[who].get(d) is not None]
            scores[f"{who}_{DIR_LABEL[d]}"] = {
                "micro": _rate(num, den),
                "macro": (sum(vals) / len(vals)) if vals else None,
            }

    # (B) Agreement, and (C) judge-as-classifier with the human as gold.
    def _confusion(subset: list[dict]) -> dict:
        tp = sum(1 for r in subset if r["human"] and r["judge"])
        fp = sum(1 for r in subset if not r["human"] and r["judge"])
        fn = sum(1 for r in subset if r["human"] and not r["judge"])
        tn = sum(1 for r in subset if not r["human"] and not r["judge"])
        prec = _rate(tp, tp + fp)
        rec_ = _rate(tp, tp + fn)
        f1   = (2 * prec * rec_ / (prec + rec_)) if prec and rec_ else None
        return {"n": len(subset), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "agreement": _rate(tp + tn, len(subset)), "kappa": _cohens_kappa(tp, fp, fn, tn),
                "present_precision": prec, "present_recall": rec_, "present_f1": f1}

    agreement = {"overall": _confusion(rows)}
    for d in DIRECTIONS:
        agreement[DIR_LABEL[d]] = _confusion([r for r in rows if r["direction"] == DIR_LABEL[d]])

    # Macro: agreement computed per record, then averaged — matches how the score
    # columns are macro-averaged, so a long record can't dominate a short one.
    # PABAK (2*po - 1) is the chance-corrected companion; per-record kappa is not
    # reported because single records have degenerate margins far too often.
    def _macro(pairs: list[tuple[int, int]]) -> tuple[float | None, float | None, int]:
        rates = [a / n for a, n in pairs if n]
        if not rates:
            return None, None, 0
        po = sum(rates) / len(rates)
        return po, 2 * po - 1, len(rates)

    for name, dirs in (("overall", DIRECTIONS), ("recall", ("r",)), ("precision", ("p",))):
        pairs = [(sum(rec["dir_agree"][d][0] for d in dirs if d in rec["dir_agree"]),
                  sum(rec["dir_agree"][d][1] for d in dirs if d in rec["dir_agree"]))
                 for rec in per_record]
        po, pabak, n_rec = _macro(pairs)
        agreement[name].update({"macro_agreement": po, "macro_pabak": pabak,
                                "macro_n_records": n_rec})

    graded_records = [rec for rec in per_record if rec["n_graded"]]
    exact = _rate(sum(1 for rec in graded_records if rec["exact_match"]), len(graded_records))

    return {"rows": rows, "per_record": per_record, "scores": scores,
            "agreement": agreement, "exact_match": exact,
            "n_records": len(records), "n_graded_records": len(graded_records),
            "errors": errors}


def report(stats: dict, args) -> None:
    s, ag = stats["scores"], stats["agreement"]
    ov = ag["overall"]

    print(f"\n{BOLD}── Scores as each rater would report them ──────────────────{RESET}")
    print(f"{DIM}recall = share of REFERENCE facts judged entailed; "
          f"precision = share of ANSWER facts judged supported{RESET}")
    print(f"  {'':10} {'you (micro)':>12} {'judge (micro)':>14} {'Δ':>8}   "
          f"{'you (macro)':>12} {'judge (macro)':>14} {'Δ':>8}")
    for name in ("recall", "precision"):
        h, j = s[f"human_{name}"], s[f"judge_{name}"]
        d_mi = (j["micro"] - h["micro"]) if (h["micro"] is not None and j["micro"] is not None) else None
        d_ma = (j["macro"] - h["macro"]) if (h["macro"] is not None and j["macro"] is not None) else None
        print(f"  {name:10} {_fmt(h['micro']):>12} {_fmt(j['micro']):>14} {_fmt(d_mi):>8}   "
              f"{_fmt(h['macro']):>12} {_fmt(j['macro']):>14} {_fmt(d_ma):>8}")
    print(f"  {DIM}Δ > 0 means the judge is more lenient than you (calls more facts Present).{RESET}")

    print(f"\n{BOLD}── Agreement (judge vs you, per fact) ──────────────────────{RESET}")
    print(f"  {'':12} {'n':>6} {'agree':>8} {'kappa':>8}   {'agree (macro)':>14} {'pabak (macro)':>14}")
    for name in ("overall", "recall", "precision"):
        a = ag[name]
        print(f"  {name:12} {a['n']:>6} {_fmt(a['agreement']):>8} {_fmt(a['kappa'], pct=False):>8}   "
              f"{_fmt(a['macro_agreement']):>14} {_fmt(a['macro_pabak'], pct=False):>14}")
    print(f"  {DIM}macro = computed per record, then averaged over the "
          f"{ag['overall']['macro_n_records']} records{RESET}")
    em = stats["exact_match"]
    print(f"  exact match: {_fmt(em)} of records "
          f"({sum(1 for r in stats['per_record'] if r['n_graded'] and r['exact_match'])}"
          f"/{stats['n_graded_records']} had every fact matching)")

    print(f"\n{BOLD}── Judge as a classifier of \"Present\" (you = gold) ─────────{RESET}")
    print(f"  precision {_fmt(ov['present_precision'])}   recall {_fmt(ov['present_recall'])}   "
          f"F1 {_fmt(ov['present_f1'])}")
    print(f"  {DIM}TP {ov['tp']}   FP {ov['fp']} (judge Present, you Absent)   "
          f"FN {ov['fn']} (judge Absent, you Present)   TN {ov['tn']}{RESET}")

    disagreements = [r for r in stats["rows"] if not r["agree"]]
    if disagreements:
        shown = disagreements if args.max_show <= 0 else disagreements[: args.max_show]
        print(f"\n{BOLD}── Disagreements ({len(disagreements)}) ───────────────────────────────{RESET}")
        for r in shown:
            you   = f"{GREEN}Present{RESET}" if r["human"] else f"{RED}Absent{RESET}"
            judge = f"{GREEN}Present{RESET}" if r["judge"] else f"{RED}Absent{RESET}"
            print(f"  card {r['card']:<3} {r['direction']}[{r['idx']}]  you={you} judge={judge}"
                  f"  {DIM}{r['record_id']}{RESET}")
            print(f"      {DIM}{r['fact']}{RESET}")
        if len(disagreements) > len(shown):
            print(f"  {DIM}... {len(disagreements) - len(shown)} more (--max-show 0 for all,"
                  f" or --out to dump every fact){RESET}")

    for e in stats["errors"]:
        print(f"{YELLOW}! {e}{RESET}")


def _label(v) -> str:
    return "" if v is None else ("Present" if v else "Absent")


def write_csv(stats: dict, path: str) -> None:
    fields = ["card", "card_index", "record_id", "question_id", "source_name", "direction",
              "idx", "human", "judge", "agree", "fact", "your_note"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(stats["rows"])
    print(f"Per-fact detail -> {path}")


def write_disagreements(stats: dict, path: str) -> None:
    """Review sheet: one row per disagreement, ordered by UI card so you can walk the
    annotation page top to bottom. `revised_label` / `comment` are blank for you to fill in."""
    fields = ["card", "card_index", "direction", "idx", "fact",
              "your_label", "judge_label", "your_note", "revised_label", "comment",
              "record_id", "question_id", "source_name"]
    rows = sorted((r for r in stats["rows"] if not r["agree"]),
                  key=lambda r: (r["card_index"], r["direction"], r["idx"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "your_label": _label(r["human"]), "judge_label": _label(r["judge"]),
                        "revised_label": "", "comment": ""})
    print(f"Disagreements ({len(rows)}) -> {path}")
    print(f"  {DIM}`card` is the number shown in the annotation sidebar; `idx` is the fact's "
          f"position within that direction's column.{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="+", required=True,
                    help="Export(s) from fact_annotation.html — file, dir, or glob")
    ap.add_argument("--model", default="gemini_flash_juror",
                    help="Alias from utils._VERTEX_GEMINI_MODELS "
                         "(default: gemini_flash_juror = gemini-3.1-flash-lite)")
    ap.add_argument("--prompts", default=None,
                    help="YAML with recall/precision/system to test INSTEAD of the live "
                         "score_facts_batch.py constants")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Run the whole sample N times and report each run (checks stability)")
    ap.add_argument("--concurrency", type=int, default=10, help="Parallel online calls (default: 10)")
    ap.add_argument("--max-show", type=int, default=15,
                    help="Disagreements printed; 0 = all (default: 15)")
    ap.add_argument("--out", default=None, help="Write per-fact detail to this CSV")
    ap.add_argument("--disagreements", default=None,
                    help="Write a review sheet of every disagreement to this CSV")
    ap.add_argument("--records", default=None,
                    help="The generated annotate.html — gives the authoritative UI card order")
    ap.add_argument("--map", default=None,
                    help="annotate.map.json — alternative source of the UI card order")
    ap.add_argument("--exclude-records", nargs="+", default=None, metavar="ID_OR_FILE",
                    help="Drop these record_ids — or every record_id found in the given "
                         "annotation export / map.json. Use to hold out the records a "
                         "prompt was tuned on so the rest is a clean generalization test.")
    args = ap.parse_args()

    records = load_records(args.annotations)
    if not records:
        sys.exit("ERROR: no annotated records loaded.")

    if args.exclude_records:
        drop: set[str] = set()
        for item in args.exclude_records:
            p = Path(item)
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                ids = data.keys() if isinstance(data, dict) else \
                    (a.get("record_id") for a in data if isinstance(a, dict))
                drop.update(i for i in ids if i)
            else:
                drop.add(item)
        before = len(records)
        records = [r for r in records if r["record_id"] not in drop]
        print(f"  Excluded {before - len(records)} record(s) via --exclude-records "
              f"({len(drop)} id(s) given); {len(records)} remain")
        if not records:
            sys.exit("ERROR: every record was excluded.")

    order = load_card_order(args.records, args.map)
    if order:
        records.sort(key=lambda r: order.get(r["record_id"], 10**6))
    else:
        print("  NOTE: no --records/--map given; card numbers follow the export order, which "
              "matches the UI only if every card was completed.")
    for i, rec in enumerate(records):
        rec["card_index"] = order.get(rec["record_id"], i)      # 0-based
        rec["card"]       = rec["card_index"] + 1               # as shown in the sidebar
    prompts = load_prompts(args.prompts)
    n_facts = sum(len(r["facts"][d]) for r in records for d in DIRECTIONS)
    n_marked = sum(1 for r in records for d in DIRECTIONS for m in r["human"][d] if m is not None)
    print(f"{len(records)} records, {n_facts} facts ({n_marked} labelled), "
          f"model={args.model}, source={'--prompts ' + args.prompts if args.prompts else 'score_facts_batch.py'}")

    for run in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n{BOLD}════ run {run}/{args.repeat} ════{RESET}")
        judged = asyncio.run(_one_pass(records, prompts, args.model, args.concurrency))
        stats  = compute(records, judged)
        report(stats, args)
        print()
        if args.out:
            out = args.out if args.repeat == 1 else f"{Path(args.out).with_suffix('')}_run{run}.csv"
            write_csv(stats, out)
        if args.disagreements:
            dis = (args.disagreements if args.repeat == 1
                   else f"{Path(args.disagreements).with_suffix('')}_run{run}.csv")
            write_disagreements(stats, dis)


if __name__ == "__main__":
    main()
