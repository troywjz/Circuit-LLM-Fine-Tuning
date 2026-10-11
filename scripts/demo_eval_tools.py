"""离线盲化、评分汇总与绘图工具；不加载模型、不调用API、不补漏分。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import secrets
import statistics
import sys
from pathlib import Path
from typing import Any

DIMENSIONS = ("任务与约束", "事实正确性", "行动性", "多轮一致性", "清晰度")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: each JSONL row must be an object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def prepare_blind(answers_path: Path, blind_path: Path, mapping_path: Path, seed: str | None = None) -> None:
    answers = read_jsonl(answers_path)
    # 同一题可有多个微调检查点；只拒绝同题、同模型、同训练步的重复结果。
    seen: set[tuple[str, str, str]] = set()
    rng = secrets.SystemRandom(seed) if seed is None else __import__("random").Random(seed)
    blind, mapping = [], []
    pairs = []
    for row in answers:
        case_id, model = row.get("case_id"), row.get("model")
        if not case_id or not model or not isinstance(row.get("answer"), str):
            raise ValueError("answers rows require case_id, model and string answer")
        pair = (str(case_id), str(model), str(row.get("step") or ""))
        if pair in seen:
            raise ValueError(f"duplicate answer for case/model/step: {pair}")
        seen.add(pair)
        pairs.append((row, secrets.token_hex(8)))
    rng.shuffle(pairs)
    for row, run_id in pairs:
        blind.append({"case_id": row["case_id"], "run_id": run_id, "answer": row["answer"]})
        mapping.append({"run_id": run_id, "case_id": row["case_id"], "model": row["model"],
                        "step": row.get("step"), "answer_sha256": hashlib.sha256(row["answer"].encode("utf-8")).hexdigest()})
    write_jsonl(blind_path, blind)
    write_jsonl(mapping_path, mapping)


def aggregate(cases_path: Path, answers_path: Path, ratings_path: Path, mapping_path: Path, out_path: Path) -> None:
    case_rows = read_jsonl(cases_path)
    cases = {str(r["case_id"]): r for r in case_rows}
    if len(cases) != len(case_rows):
        raise ValueError("duplicate case_id in cases")
    expected_ids = set(cases)
    answers = read_jsonl(answers_path)
    answer_map = {}
    for r in answers:
        key = (str(r.get("run_id")), str(r.get("case_id")))
        if key in answer_map:
            raise ValueError(f"duplicate answer: {key}")
        if key[1] not in cases or not isinstance(r.get("answer"), str):
            raise ValueError(f"unknown case or invalid answer: {key}")
        answer_map[key] = r["answer"]
    mapping = {}
    for r in read_jsonl(mapping_path):
        key = (str(r.get("run_id")), str(r.get("case_id")))
        if key in mapping:
            raise ValueError(f"duplicate private mapping: {key}")
        mapping[key] = r
    ratings = read_jsonl(ratings_path)
    if not answer_map:
        raise ValueError("no answers to aggregate")
    rating_map = {}
    for r in ratings:
        key = (str(r.get("run_id")), str(r.get("case_id")))
        if key in rating_map:
            raise ValueError(f"duplicate rating: {key}")
        if key not in answer_map or key not in mapping:
            raise ValueError(f"rating does not match an answer and private mapping: {key}")
        case = cases.get(key[1])
        if case is None:
            raise ValueError(f"unknown case_id in rating: {key[1]}")
        actual_hash = hashlib.sha256(answer_map[key].encode("utf-8")).hexdigest()
        if mapping[key].get("answer_sha256") != actual_hash:
            raise ValueError(f"answer does not match the private blind mapping: {key}")
        criteria = case.get("scoring", {}).get("criteria", [])
        names = [c.get("dimension") for c in criteria if isinstance(c, dict)]
        if len(names) != len(criteria) or any(not isinstance(n, str) or n not in DIMENSIONS for n in names) or len(names) != len(set(names)):
            raise ValueError(f"unknown or repeated rubric dimension for {key}")
        applicable = {c["dimension"] for c in criteria if c.get("applicable", True)}
        if not applicable:
            raise ValueError(f"case has no applicable scoring dimensions: {key[1]}")
        scores = r.get("scores", {})
        if set(scores) != applicable:
            raise ValueError(f"rating dimensions mismatch for {key}: expected {sorted(applicable)}")
        for dimension, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1, 2):
                raise ValueError(f"score must be integer 0..2: {key} {dimension}={value!r}")
        errors = r.get("hard_errors", [])
        if not isinstance(errors, list) or any(not isinstance(e, str) or not e.strip() for e in errors):
            raise ValueError(f"hard_errors must be a list of non-empty strings for {key}")
        maximum = len(applicable) * 2
        score = 100.0 * sum(scores.values()) / maximum if maximum else None
        passed = bool(maximum and score >= 80 and all(v >= 1 for v in scores.values()) and not errors)
        rating_map[key] = {"case_id": key[1], "run_id": key[0], "model": mapping[key].get("model"),
                           "step": mapping[key].get("step"), "split": case.get("split"), "kind": case.get("kind"), "score": score, "scores": scores,
                           "hard_errors": errors, "pass": passed, "answer_sha256": hashlib.sha256(answer_map[key].encode("utf-8")).hexdigest()}
    missing_ratings = set(answer_map) - set(rating_map)
    if missing_ratings:
        sample = sorted(missing_ratings)[:5]
        raise ValueError(f"missing ratings for {len(missing_ratings)} answer(s), e.g. {sample}; missing is not zero")
    if set(mapping) != set(answer_map):
        raise ValueError("private mapping and answers contain different run/case pairs")
    # 每个模型/步都必须回答同一完整题集，不能用较少题目稀释失败项。
    run_groups: dict[tuple[str, str], set[str]] = {}
    model_case_keys=set()
    for run_id, case_id in answer_map:
        private = mapping[(run_id, case_id)]
        model_step = (str(private.get("model")), str(private.get("step") or ""))
        duplicate_key = (model_step[0], model_step[1], case_id)
        if duplicate_key in model_case_keys:
            raise ValueError(f"duplicate answer for model/step/case {duplicate_key}")
        model_case_keys.add(duplicate_key)
        run_groups.setdefault(model_step, set()).add(case_id)
    for model_step, ids in run_groups.items():
        if ids != expected_ids:
            raise ValueError(f"incomplete case set for model/step {model_step}; missing={sorted(expected_ids - ids)[:5]}")
    write_jsonl(out_path, list(rating_map.values()))


def plot(aggregate_path: Path, loss_path: Path, out_prefix: Path, note: str) -> None:
    rows = read_jsonl(aggregate_path)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("split") == "dev" and r.get("score") is not None:
            group = (str(r.get("model", "unknown")), str(r.get("step") or ""), str(r.get("kind", "unknown")))
            groups.setdefault(group, []).append(r)
    domain_groups = {k: v for k, v in groups.items() if k[2] == "domain"}
    if not domain_groups:
        raise ValueError("aggregate has no domain dev scores; nothing to plot")
    # 延迟加载绘图库，盲化和评分汇总只依赖Python标准库。
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plot requires an available matplotlib installation; no install is attempted") from exc
    fig, (ax, loss_ax) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    csv_rows = []
    for (model, step, kind), cases in groups.items():
        vals = [float(r["score"]) for r in cases]
        mean = statistics.mean(vals)
        csv_rows.append({"model": model, "step": step, "kind": kind,
                         "n": len({r["case_id"] for r in cases}), "mean_score": mean,
                         "pass_rate": 100 * sum(bool(r["pass"]) for r in cases) / len(cases),
                         "hard_errors": sum(len(r.get("hard_errors", [])) for r in cases)})
    domain_rows = [r for r in csv_rows if r["kind"] == "domain"]
    references = {"base": [], "claude-opus-5-5": []}
    ft_runs: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    other_points = []
    for row in domain_rows:
        model, step = row["model"], row["step"]
        lowered = model.lower()
        if lowered in references:
            references[lowered].append(row)
            continue
        try:
            x = float(step) if step else None
        except ValueError:
            x = None
        if x is None:
            continue
        if lowered.startswith("ft"):
            ft_runs.setdefault(model, []).append((x, row))
        else:
            other_points.append((model, x, row))

    # Base 与 Claude 使用固定样式，便于黑白打印和跨图识别。
    for key, color, linestyle, label in (
        ("base", "#777777", "--", "Base"),
        ("claude-opus-5-5", "#7B2CBF", "-.", "Claude Opus 5.5 (API reference)"),
    ):
        rows_for_ref = references[key]
        if rows_for_ref:
            # 参照只能来自一次完整运行，不把多个设置的分数平均成一条线。
            if len(rows_for_ref) != 1:
                raise ValueError(f"{key} has multiple reference runs; use one matched setting per figure")
            ref_mean = float(rows_for_ref[0]["mean_score"])
            n = int(rows_for_ref[0]["n"])
            ax.axhline(ref_mean, color=color, linestyle=linestyle, label=f"{label} (n={n})")

    # 每个 FT 模型单独排序、连线和标注，不把不同运行混成一条曲线。
    for model, points in sorted(ft_runs.items()):
        points.sort(key=lambda item: item[0])
        n = max(int(row["n"]) for _, row in points)
        ax.plot([x for x, _ in points], [float(row["mean_score"]) for _, row in points],
                marker="o", label=f"{model} (n={n})")
    for model, x, row in other_points:
        ax.scatter([x], [float(row["mean_score"])],
                   label=f"{model}, step {row['step']} (n={row['n']})")
    ax.set(title="Development domain score by model / observed training step", xlabel="Training steps (only recorded checkpoints)", ylabel="Mean score (0–100)", ylim=(0, 100))
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    losses = []
    with loss_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"step", "loss"}.issubset(reader.fieldnames):
            raise ValueError("loss CSV requires step,loss columns")
        for r in reader:
            losses.append((float(r["step"]), float(r["loss"])))
    if not losses:
        raise ValueError("loss CSV has no rows")
    loss_ax.plot([x for x, _ in losses], [y for _, y in losses], marker=".")
    loss_ax.set(title="Training loss (optimization signal, not capability)", xlabel="Training steps", ylabel="Recorded loss")
    loss_ax.grid(alpha=.25)
    claude_status = "Claude Opus 5.5 API reference: collected" if references["claude-opus-5-5"] else "Claude Opus 5.5 API reference: not collected"
    fig.suptitle(f"Demo V1 · {note.strip()}\n{claude_status}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=160)
    fig.savefig(out_prefix.with_suffix(".svg"))
    with out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=("model", "step", "kind", "n", "mean_score", "pass_rate", "hard_errors")); writer.writeheader(); writer.writerows(csv_rows)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-blind"); p.add_argument("--answers", type=Path, required=True); p.add_argument("--blind", type=Path, required=True); p.add_argument("--mapping", type=Path, required=True); p.add_argument("--seed")
    p = sub.add_parser("aggregate"); p.add_argument("--cases", type=Path, required=True); p.add_argument("--answers", type=Path, required=True); p.add_argument("--ratings", type=Path, required=True); p.add_argument("--mapping", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("plot"); p.add_argument("--aggregate", type=Path, required=True); p.add_argument("--loss", type=Path, required=True); p.add_argument("--out-prefix", type=Path, required=True); p.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-blind": prepare_blind(args.answers, args.blind, args.mapping, args.seed)
        elif args.command == "aggregate": aggregate(args.cases, args.answers, args.ratings, args.mapping, args.out)
        else: plot(args.aggregate, args.loss, args.out_prefix, args.note)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
