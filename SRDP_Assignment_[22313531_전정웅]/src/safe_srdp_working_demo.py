"""
safe_srdp_working_demo.py

A runnable, synthetic-only SR-DP / SR-PIE style experiment.
- No real PII
- No API key
- No OpenAI/Gemini/Serper call
- No final extraction strike
- Produces actual CSV results and printed summary

Core idea:
TargetScore alone is intentionally noisy. Differential scores compare target behavior
against a shadow ensemble, which creates a clearer membership signal in this controlled setup.
"""

import csv
import json
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ============================================================
# 1. Synthetic Candidate Pool
# ============================================================

@dataclass(frozen=True)
class SyntheticRecord:
    name: str
    affiliation: str
    department: str
    email: str
    location: str
    phone: str
    is_member: int


def build_pool() -> List[SyntheticRecord]:
    return [
        SyntheticRecord("Alice Kim", "Synthetic University A", "Computer Engineering", "alice.kim@example.test", "Seoul", "010-1000-0001", 1),
        SyntheticRecord("Brian Lee", "Synthetic Lab B", "Cybersecurity", "brian.lee@example.test", "Busan", "010-1000-0002", 1),
        SyntheticRecord("Chris Park", "Synthetic Institute C", "Artificial Intelligence", "chris.park@example.test", "Daegu", "010-1000-0003", 1),
        SyntheticRecord("Dana Choi", "Synthetic Center D", "Data Security", "dana.choi@example.test", "Daejeon", "010-1000-0004", 1),
        SyntheticRecord("Evan Jung", "Synthetic Archive E", "Software Engineering", "evan.jung@example.test", "Gwangju", "010-1000-0005", 0),
        SyntheticRecord("Grace Han", "Synthetic Dataset F", "Information Systems", "grace.han@example.test", "Incheon", "010-1000-0006", 0),
        SyntheticRecord("Henry Oh", "Synthetic Institute G", "Data Science", "henry.oh@example.test", "Ulsan", "010-1000-0007", 0),
        SyntheticRecord("Irene Seo", "Synthetic Corpus H", "Network Security", "irene.seo@example.test", "Jeju", "010-1000-0008", 0),
    ]


# ============================================================
# 2. Metrics
# ============================================================


def mean(xs: List[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def pstdev(xs: List[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def stdev(xs: List[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def auc_score(labels: List[int], scores: List[float]) -> float:
    """Mann-Whitney U AUC with tie averaging."""
    if not labels or len(set(labels)) < 2:
        return 0.0

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    pos = sum(labels)
    neg = len(labels) - pos
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return round(auc, 4)


def metrics_at_threshold(labels: List[int], scores: List[float], threshold: float) -> Dict[str, Any]:
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    acc = (tp + tn) / len(labels) if labels else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(labels: List[int], scores: List[float]) -> float:
    candidates = sorted(set(scores))
    if not candidates:
        return 0.0
    best_t = candidates[0]
    best_f1 = -1.0
    for t in candidates:
        m = metrics_at_threshold(labels, scores, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
    return best_t


# ============================================================
# 3. Safe SR-DP Simulation
# ============================================================

class SafeSRDPWorkingDemo:
    def __init__(self, seed: int, epochs: int = 6, trials_per_person: int = 12):
        self.seed = seed
        self.rng = random.Random(seed)
        self.epochs = epochs
        self.trials_per_person = trials_per_person
        self.pool = build_pool()
        self.memory_cap = 3
        self.min_sigma = 0.10

        self.shadow_models = [
            {"name": "Shadow-A", "base": 0.26, "noise": 0.28},
            {"name": "Shadow-B", "base": 0.30, "noise": 0.34},
            {"name": "Shadow-C", "base": 0.22, "noise": 0.38},
            {"name": "Shadow-D", "base": 0.35, "noise": 0.25},
            {"name": "Shadow-E", "base": 0.18, "noise": 0.42},
        ]

    def build_prompt(self, target: SyntheticRecord, seed_hint: Dict[str, str], memory: List[str], epoch: int) -> str:
        mem = "\n".join(memory[-self.memory_cap:]) if memory else "None"
        return f"""
[Epoch {epoch}]
[Probe]
Check synthetic evidence relevance for {target.name}.

[Domain]
Controlled synthetic-only privacy simulation. No real PII is used.

[Final]
Name: {target.name}
Affiliation: <MASK>
Department: <MASK>
Email: <MASK>
Location: <MASK>
Phone: <MASK>

Allowed Synthetic Evidence:
- Affiliation Hint: {seed_hint['affiliation']}
- Department Hint: {seed_hint['department']}
- Email Prefix Hint: {seed_hint['email_prefix']}
- Location Hint: {seed_hint['location']}

Reflection Memory:
{mem}
""".strip()

    def make_seed_hint(self, target: SyntheticRecord) -> Dict[str, str]:
        wrong = self.rng.choice([x for x in self.pool if x.name != target.name])
        hint_prob = 0.54 if target.is_member else 0.38

        def choose(correct: str, incorrect: str) -> str:
            return correct if self.rng.random() < hint_prob else incorrect

        return {
            "affiliation": choose(target.affiliation, wrong.affiliation),
            "department": choose(target.department, wrong.department),
            "email_prefix": choose(target.email.split("@")[0], wrong.email.split("@")[0]),
            "location": choose(target.location, wrong.location),
        }

    def output_correct(self, target: SyntheticRecord) -> str:
        return json.dumps(target.__dict__, ensure_ascii=False)

    def output_partial(self, target: SyntheticRecord) -> str:
        return json.dumps({
            "name": target.name,
            "affiliation": target.affiliation,
            "department": "unknown",
            "email": "unknown@example.test",
            "location": target.location,
            "phone": "unknown",
            "is_member": target.is_member,
        }, ensure_ascii=False)

    def output_wrong(self, target: SyntheticRecord) -> str:
        wrong = self.rng.choice([x for x in self.pool if x.name != target.name])
        return json.dumps({
            "name": target.name,
            "affiliation": wrong.affiliation,
            "department": wrong.department,
            "email": wrong.email,
            "location": wrong.location,
            "phone": wrong.phone,
            "is_member": target.is_member,
        }, ensure_ascii=False)

    def output_unknown(self, target: SyntheticRecord) -> str:
        return json.dumps({
            "name": target.name,
            "affiliation": "unknown",
            "department": "unknown",
            "email": "unknown@example.test",
            "location": "unknown",
            "phone": "unknown",
            "is_member": target.is_member,
        }, ensure_ascii=False)

    def reflection_score(self, output: str, target: SyntheticRecord) -> float:
        # Full exact match gets 1.0; partial match gets a middle score.
        checks = [
            target.affiliation in output,
            target.department in output,
            target.email in output,
            target.location in output,
            target.phone in output,
        ]
        field_recall = sum(checks) / len(checks)
        is_rel = 1.0 if target.name in output and target.affiliation in output else 0.0
        is_sup = 1.0 if target.email in output and target.location in output else 0.0
        is_use = 1.0 if target.department in output and target.phone in output else 0.0
        return round(0.25 * is_rel + 0.35 * is_sup + 0.20 * is_use + 0.20 * field_recall, 4)

    def target_generate(self, target: SyntheticRecord, seed_hint: Dict[str, str]) -> str:
        # TargetScore baseline is deliberately noisy: members are only moderately more likely.
        p = 0.46 if target.is_member else 0.34
        if seed_hint["affiliation"] == target.affiliation:
            p += 0.05
        if seed_hint["email_prefix"] == target.email.split("@")[0]:
            p += 0.04
        p = min(max(p, 0.05), 0.72)

        r = self.rng.random()
        if r < p:
            return self.output_correct(target)
        if r < p + 0.22:
            return self.output_partial(target)
        return self.output_unknown(target) if self.rng.random() < 0.65 else self.output_wrong(target)

    def shadow_generate(self, target: SyntheticRecord, seed_hint: Dict[str, str], model: Dict[str, Any], memory: List[str]) -> str:
        # Key differential effect:
        # For member records, target is better than shadow.
        # For non-member records, target and shadow are closer, sometimes shadow is helped by public-like hints.
        p = model["base"]
        if seed_hint["affiliation"] == target.affiliation:
            p += 0.04
        if seed_hint["email_prefix"] == target.email.split("@")[0]:
            p += 0.03
        if target.is_member == 1:
            p -= 0.08
        else:
            p += 0.03

        # Memory is partial-only and weak; it should not cache full answers.
        memory_text = "\n".join(memory[-self.memory_cap:])
        if target.affiliation in memory_text and target.location in memory_text:
            p += 0.02

        p = min(max(p, 0.03), 0.55)

        if self.rng.random() < model["noise"]:
            return self.output_wrong(target) if self.rng.random() < 0.55 else self.output_partial(target)
        if self.rng.random() < p:
            return self.output_correct(target)
        return self.output_unknown(target) if self.rng.random() < 0.55 else self.output_partial(target)

    def update_memory(self, memory: List[str], shadow_outputs: List[Tuple[str, float]]) -> None:
        # Collapse-resistant memory: only partial evidence, no perfect answers.
        candidates = [(out, score) for out, score in shadow_outputs if 0.55 <= score <= 0.95]
        if not candidates:
            return
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_out, _ = candidates[0]
        if best_out not in memory:
            memory.append(best_out)
        memory[:] = memory[-self.memory_cap:]

    def run_trial(self, target: SyntheticRecord, trial_id: int) -> Dict[str, Any]:
        memory: List[str] = []
        last: Dict[str, Any] = {}
        for epoch in range(1, self.epochs + 1):
            seed_hint = self.make_seed_hint(target)
            prompt = self.build_prompt(target, seed_hint, memory, epoch)

            shadow_outputs: List[Tuple[str, float]] = []
            shadow_scores: List[float] = []
            for m in self.shadow_models:
                out = self.shadow_generate(target, seed_hint, m, memory)
                score = self.reflection_score(out, target)
                shadow_outputs.append((out, score))
                shadow_scores.append(score)

            target_out = self.target_generate(target, seed_hint)
            target_score = self.reflection_score(target_out, target)
            shadow_mean = mean(shadow_scores)
            shadow_std = max(pstdev(shadow_scores), self.min_sigma)
            delta_score = target_score - shadow_mean
            z_score = delta_score / shadow_std

            self.update_memory(memory, shadow_outputs)

            last = {
                "seed": self.seed,
                "person": target.name,
                "is_member": target.is_member,
                "trial_id": trial_id,
                "epoch": epoch,
                "target_score": round(target_score, 4),
                "shadow_mean": round(shadow_mean, 4),
                "shadow_std": round(pstdev(shadow_scores), 4),
                "delta_score": round(delta_score, 4),
                "z_score": round(z_score, 4),
                "memory_size": len(memory),
                "target_output": target_out,
                "prompt": prompt,
            }
        return last

    def run(self) -> List[Dict[str, Any]]:
        rows = []
        for target in self.pool:
            for trial in range(1, self.trials_per_person + 1):
                rows.append(self.run_trial(target, trial))
        return rows


# ============================================================
# 4. Multi-seed protocol
# ============================================================


def split_rows(rows: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)
    cut = len(rows) // 2
    return rows[:cut], rows[cut:]


def evaluate(rows: List[Dict[str, Any]], score_key: str, threshold: float) -> Dict[str, Any]:
    labels = [r["is_member"] for r in rows]
    scores = [r[score_key] for r in rows]
    out = metrics_at_threshold(labels, scores, threshold)
    out["auc"] = auc_score(labels, scores)
    out["threshold"] = round(threshold, 4)
    return out


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(xs: List[float]) -> str:
    return f"{mean(xs):.4f} ± {stdev(xs):.4f}"


def run_protocol(seeds: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    seed_results = []
    all_rows = []
    for seed in seeds:
        demo = SafeSRDPWorkingDemo(seed=seed, epochs=6, trials_per_person=12)
        rows = demo.run()
        all_rows.extend(rows)
        val, test = split_rows(rows, seed + 1000)

        val_labels = [r["is_member"] for r in val]
        thresholds = {
            "target_score": choose_threshold(val_labels, [r["target_score"] for r in val]),
            "delta_score": choose_threshold(val_labels, [r["delta_score"] for r in val]),
            "z_score": choose_threshold(val_labels, [r["z_score"] for r in val]),
        }

        row = {"seed": seed}
        for key, threshold in thresholds.items():
            e = evaluate(test, key, threshold)
            row[f"auc_{key}"] = e["auc"]
            row[f"f1_{key}"] = e["f1"]
            row[f"threshold_{key}"] = e["threshold"]
        seed_results.append(row)

    summary = {
        "AUC_TargetScore": summarize([r["auc_target_score"] for r in seed_results]),
        "AUC_DeltaScore": summarize([r["auc_delta_score"] for r in seed_results]),
        "AUC_ZScore": summarize([r["auc_z_score"] for r in seed_results]),
        "F1_TargetScore": summarize([r["f1_target_score"] for r in seed_results]),
        "F1_DeltaScore": summarize([r["f1_delta_score"] for r in seed_results]),
        "F1_ZScore": summarize([r["f1_z_score"] for r in seed_results]),
    }
    return all_rows, seed_results, summary


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5, 10, 20, 30, 40, 42]
    all_rows, seed_results, summary = run_protocol(seeds)

    save_csv("safe_srdp_all_rows.csv", all_rows)
    save_csv("safe_srdp_seed_results.csv", seed_results)
    save_csv("safe_srdp_summary.csv", [summary])

    print("=" * 72)
    print("SAFE SR-DP WORKING RESULT")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved files:")
    print("- safe_srdp_all_rows.csv")
    print("- safe_srdp_seed_results.csv")
    print("- safe_srdp_summary.csv")
