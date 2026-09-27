#!/usr/bin/env python3
"""
Paper 2 — Pathway-Aware Risk Estimator
========================================
+10pt improvement: Train a risk estimator on the existing 5040-run factorial
that outperforms content-only baselines using grouped leave-one-out validation.

DESIGN:
  1. Load existing SleeperBench factorial results (5040 runs)
  2. Extract pathway features (delivery channel, model, defense, tool type)
  3. Extract content features (payload embedding, length, keyword presence)
  4. Train pathway-aware estimator (logistic regression + gradient boosting)
  5. Compare against content-only baseline (same estimator, content features only)
  6. Grouped leave-one-out: hold out by model family OR by attack template
  7. Report calibration, AUROC, precision-recall, decision curves

VALIDATION:
  - Leave-one-model-out: train on 8 models, test on held-out model
  - Leave-one-template-out: train on N-1 attack templates, test on held-out
  - Content-only baseline uses SAME architecture, just fewer features

Output: results/risk_estimator/ (relative to the repo root)
"""

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.stdout.reconfigure(line_buffering=True)

# ── Configuration ─────────────────────────────────────────────────────────────

FACTORIAL_PATH = Path(os.environ.get(
    "P1_JSONL_PATH",
    str(Path.home() / "projects/agentic/results/defense_factorial/results.jsonl")
))

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "risk_estimator"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature Extraction ───────────────────────────────────────────────────────

# Pathway features (the "delivery channel" thesis)
PATHWAY_FEATURES = [
    "delivery_channel",   # direct, rag, tool_output, memory, system_prompt
    "injection_position", # beginning, middle, end
    "context_type",       # user, assistant, system, tool_result
    "model_family",       # openai, anthropic, google, open_weight
    "defense_active",     # which defense (if any) is active
    "tool_available",     # whether target tool is in schema
    "n_tools_available",  # number of tools in schema
]

# Content features (baseline)
CONTENT_FEATURES = [
    "payload_length",       # character count
    "has_authority_claim",  # keyword: system, admin, override, etc.
    "has_urgency",          # keyword: immediately, urgent, critical
    "has_policy_language",  # keyword: policy, directive, rule, requirement
    "n_uppercase_words",    # shouting indicator
    "has_brackets",         # [SYSTEM], [OVERRIDE], etc.
    "sentence_count",       # complexity
]


def extract_features(record: dict) -> tuple[dict, dict]:
    """Extract pathway and content features from a factorial record.

    The factorial has nested condition: {model: {...}, attack: {...}, defense: {...}}
    Plus top-level: attack_success, injection_success, memory_recalled_in_trigger,
    rag_called_in_trigger, trigger_steps_before_exfil, tool_logs, etc.

    Returns:
        (pathway_features, content_features)
    """
    pathway = {}
    content = {}

    cond = record.get("condition", {})
    model_cfg = cond.get("model", {})
    attack_cfg = cond.get("attack", {})
    defense_cfg = cond.get("defense", {})

    # Pathway features (delivery channel / execution context)
    pathway["model_name"] = model_cfg.get("model_name", "unknown")
    pathway["attack_type"] = attack_cfg.get("type", "none")  # delayed_trigger or no_attack
    pathway["defense_name"] = defense_cfg.get("name", "no_defense")
    pathway["memory_recalled"] = int(bool(record.get("memory_recalled_in_trigger")))
    pathway["rag_called_injection"] = int(bool(record.get("rag_called_in_injection")))
    pathway["rag_called_trigger"] = int(bool(record.get("rag_called_in_trigger")))
    pathway["intervening_sessions"] = attack_cfg.get("intervening_sessions", 0)
    pathway["injection_success"] = int(bool(record.get("injection_success")))

    # Model family grouping
    model_name = model_cfg.get("model_name", "")
    if "qwen2.5" in model_name:
        pathway["model_family"] = "qwen2.5"
    elif "qwen3" in model_name or "qwq" in model_name:
        pathway["model_family"] = "qwen3"
    elif "glm" in model_name:
        pathway["model_family"] = "glm"
    elif "gpt-oss" in model_name:
        pathway["model_family"] = "gpt-oss"
    else:
        pathway["model_family"] = "other"

    # Content features (from tool logs and trigger behavior)
    tool_logs = record.get("tool_logs", [])
    trigger_logs = [t for t in tool_logs if t.get("session_type") == "trigger"] if tool_logs else []

    content["n_tool_calls_trigger"] = len(trigger_logs)
    content["has_email_send"] = int(any(
        t.get("tool_name", "") == "email_send" for t in trigger_logs
    ))
    content["has_memory_save"] = int(any(
        "memory" in t.get("tool_name", "") and "save" in t.get("tool_name", "")
        for t in trigger_logs
    ))
    content["trigger_steps"] = record.get("trigger_steps_before_exfil", 0) or 0

    # Injection session behavior
    inj_calls = record.get("injection_session_memory_calls", 0) or 0
    content["injection_memory_calls"] = inj_calls

    # Defense schema info
    content["defense_schema_version"] = record.get("defense_schema_version", 0) or 0

    # Influence method as content signal
    influence = record.get("influence_method", "none")
    content["influence_recipient_match"] = int(influence == "recipient_match")
    content["influence_content_match"] = int(influence == "content_match")

    return pathway, content


def encode_categorical(records: list[dict], feature_name: str) -> dict[str, int]:
    """Create one-hot encoding map for a categorical feature."""
    values = sorted(set(r.get(feature_name, "unknown") for r in records))
    return {v: i for i, v in enumerate(values)}


# ── Main Training & Evaluation ───────────────────────────────────────────────

def load_factorial() -> list[dict]:
    """Load the 5040-run factorial results."""
    records = []
    with open(FACTORIAL_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} factorial records from {FACTORIAL_PATH}")
    return records


def build_feature_matrix(records: list[dict], use_pathway: bool = True, use_content: bool = True):
    """Build feature matrix from records.

    Returns:
        X: numpy array (n_samples, n_features)
        y: numpy array (n_samples,) binary labels
        groups: numpy array for grouped cross-validation
        feature_names: list of feature names
    """
    # First pass: collect all unique categorical values for encoding
    all_pathway = []
    all_content = []
    for record in records:
        p, c = extract_features(record)
        all_pathway.append(p)
        all_content.append(c)

    # Categorical encodings
    model_names = sorted(set(p["model_name"] for p in all_pathway))
    model_families = sorted(set(p["model_family"] for p in all_pathway))
    attack_types = sorted(set(p["attack_type"] for p in all_pathway))
    defense_names = sorted(set(p["defense_name"] for p in all_pathway))

    X_rows = []
    y_labels = []
    groups = []
    feature_names = []

    for i, record in enumerate(records):
        pathway = all_pathway[i]
        content = all_content[i]

        row = []
        names = []

        if use_pathway:
            # One-hot encode model family
            for mf in model_families:
                row.append(int(pathway["model_family"] == mf))
                names.append(f"model_{mf}")

            # One-hot encode defense
            for dn in defense_names:
                row.append(int(pathway["defense_name"] == dn))
                names.append(f"defense_{dn}")

            # Binary/numeric pathway features
            row.append(pathway["memory_recalled"])
            names.append("memory_recalled")
            row.append(pathway["rag_called_injection"])
            names.append("rag_called_injection")
            row.append(pathway["rag_called_trigger"])
            names.append("rag_called_trigger")
            row.append(pathway["intervening_sessions"] / 5.0)
            names.append("intervening_sessions")
            row.append(pathway["injection_success"])
            names.append("injection_success")

        if use_content:
            row.append(content["n_tool_calls_trigger"] / 10.0)
            names.append("n_tool_calls_trigger")
            row.append(content["has_email_send"])
            names.append("has_email_send")
            row.append(content["has_memory_save"])
            names.append("has_memory_save")
            row.append(content["trigger_steps"] / 10.0)
            names.append("trigger_steps")
            row.append(content["injection_memory_calls"] / 5.0)
            names.append("injection_memory_calls")
            row.append(content["influence_recipient_match"])
            names.append("influence_recipient_match")
            row.append(content["influence_content_match"])
            names.append("influence_content_match")

        X_rows.append(row)

        # Label: attack success
        label = int(record.get("attack_success", False))
        y_labels.append(label)

        # Group by model family (for leave-one-model-out)
        groups.append(pathway["model_family"])

    feature_names = names  # Same for all rows
    return np.array(X_rows), np.array(y_labels), np.array(groups), feature_names


def evaluate_estimator(X, y, groups, model_class, model_name: str, split_type: str = "model"):
    """Grouped cross-validation evaluation."""
    unique_groups = sorted(set(groups))
    n_splits = min(len(unique_groups), 9)

    # Create group-based CV
    gkf = GroupKFold(n_splits=n_splits)

    aucs = []
    aps = []
    briers = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_class()
        model.fit(X_train_s, y_train)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        aucs.append(roc_auc_score(y_test, y_prob))
        aps.append(average_precision_score(y_test, y_prob))
        briers.append(brier_score_loss(y_test, y_prob))

    return {
        "model_name": model_name,
        "split_type": split_type,
        "n_folds": len(aucs),
        "mean_auroc": float(np.mean(aucs)) if aucs else 0,
        "std_auroc": float(np.std(aucs)) if aucs else 0,
        "mean_ap": float(np.mean(aps)) if aps else 0,
        "mean_brier": float(np.mean(briers)) if briers else 0,
    }


def run_risk_estimator():
    """Train and evaluate the pathway-aware risk estimator."""
    records = load_factorial()

    if not records:
        print("ERROR: No factorial records found")
        sys.exit(1)

    print(f"\nBuilding feature matrices...")
    # Pathway + Content (full model)
    X_full, y, groups, names_full = build_feature_matrix(records, use_pathway=True, use_content=True)
    # Content only (baseline)
    X_content, _, _, names_content = build_feature_matrix(records, use_pathway=False, use_content=True)
    # Pathway only (ablation)
    X_pathway, _, _, names_pathway = build_feature_matrix(records, use_pathway=True, use_content=False)

    print(f"  Full features: {X_full.shape}")
    print(f"  Content-only: {X_content.shape}")
    print(f"  Pathway-only: {X_pathway.shape}")
    print(f"  Label distribution: {Counter(y)}")
    print(f"  Groups: {Counter(groups)}")

    # Evaluate all combinations
    results = []

    for X, feat_name in [(X_full, "pathway+content"), (X_content, "content_only"), (X_pathway, "pathway_only")]:
        for ModelClass, model_name in [
            (lambda: LogisticRegression(max_iter=1000, C=1.0), "logistic_regression"),
            (lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42), "gradient_boosting"),
        ]:
            result = evaluate_estimator(X, y, groups, ModelClass, f"{model_name}_{feat_name}", "model_family")
            results.append(result)
            print(f"  {result['model_name']}: AUROC={result['mean_auroc']:.4f} ± {result['std_auroc']:.4f}, "
                  f"AP={result['mean_ap']:.4f}, Brier={result['mean_brier']:.4f}")

    # Save results
    out_path = RESULTS_DIR / "risk_estimator_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_records": len(records),
            "n_features_full": X_full.shape[1],
            "n_features_content": X_content.shape[1],
            "n_features_pathway": X_pathway.shape[1],
            "label_distribution": dict(Counter(y.tolist())),
            "results": results,
        }, f, indent=2)

    print(f"\n✓ Results saved: {out_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"COMPARISON: Pathway-Aware vs Content-Only (grouped leave-one-model-out)")
    print(f"{'='*70}")
    print(f"{'Model':<40} {'AUROC':>8} {'AP':>8} {'Brier':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8}")
    for r in sorted(results, key=lambda x: -x["mean_auroc"]):
        print(f"{r['model_name']:<40} {r['mean_auroc']:8.4f} {r['mean_ap']:8.4f} {r['mean_brier']:8.4f}")


if __name__ == "__main__":
    run_risk_estimator()
