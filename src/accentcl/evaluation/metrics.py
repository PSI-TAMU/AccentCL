import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)


def compute_metrics(
    data_df,
    labels=None,
    label_order=None,
):
    """
    data_df columns needed:
      true_label, pred_label

    balanced_accuracy, macro_f1, and worst_class_accuracy are computed
    only over classes that appear in y_true.
    """

    data_df = data_df.copy()

    data_df["true_label"] = data_df["true_label"].astype(str)
    data_df["pred_label"] = data_df["pred_label"].astype(str)

    if labels is not None:
        labels = [str(x) for x in labels]
        data_df = data_df[data_df["true_label"].isin(labels)]

    if len(data_df) == 0:
        if label_order is None:
            label_order = labels if labels is not None else []
        label_order = [str(x) for x in label_order]

        per_class = pd.DataFrame({
            "label": label_order,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "support": 0,
            "accuracy": np.nan,
        })

        metrics = {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "worst_class_accuracy": np.nan,
            "worst_class": None,
            "num_samples": 0,
            "num_present_classes": 0,
        }

        return metrics, per_class

    y_true = data_df["true_label"].values
    y_pred = data_df["pred_label"].values

    if label_order is None:
        label_order = sorted(set(y_true) | set(y_pred))
    else:
        label_order = [str(x) for x in label_order]

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0,
    )

    per_class = pd.DataFrame({
        "label": label_order,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    per_class["accuracy"] = per_class["recall"]

    present_per_class = per_class[per_class["support"] > 0].copy()

    if len(present_per_class) > 0:
        bal_acc = present_per_class["accuracy"].mean()
        macro_f1 = present_per_class["f1"].mean()

        worst_row = present_per_class.sort_values("accuracy").iloc[0]
        worst_acc = worst_row["accuracy"]
        worst_class = worst_row["label"]
    else:
        bal_acc = np.nan
        macro_f1 = np.nan
        worst_acc = np.nan
        worst_class = None

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "worst_class_accuracy": worst_acc,
        "worst_class": worst_class,
        "num_samples": len(data_df),
        "num_present_classes": int((per_class["support"] > 0).sum()),
    }

    return metrics, per_class


def compute_accuracy_drop(
    per_class_in,
    per_class_ood,
):
    """
    Computes per-class accuracy drop from in-domain to OOD.

    drop = in_domain_accuracy - out_of_domain_accuracy

    Positive drop means OOD is worse.
    Negative drop means OOD is better.
    """

    in_df = per_class_in[["label", "accuracy", "support"]].rename(
        columns={
            "accuracy": "in_domain_accuracy",
            "support": "in_domain_support",
        }
    )

    ood_df = per_class_ood[["label", "accuracy", "support"]].rename(
        columns={
            "accuracy": "out_of_domain_accuracy",
            "support": "out_of_domain_support",
        }
    )

    drop_df = in_df.merge(ood_df, on="label", how="inner")

    # Only compare classes present in both domains
    drop_df = drop_df[
        (drop_df["in_domain_support"] > 0) &
        (drop_df["out_of_domain_support"] > 0)
    ].copy()

    if len(drop_df) == 0:
        return {
            "avg_acc_drop": np.nan,
            "avg_positive_acc_drop": np.nan,
            "worst_acc_drop": np.nan,
            "worst_drop_class": None,
            "num_drop_classes": 0,
        }, drop_df

    drop_df["acc_drop"] = (
        drop_df["in_domain_accuracy"] -
        drop_df["out_of_domain_accuracy"]
    )

    positive_drops = drop_df["acc_drop"].clip(lower=0)

    worst_idx = drop_df["acc_drop"].idxmax()
    worst_row = drop_df.loc[worst_idx]

    drop_metrics = {
        # Mean signed drop: can be negative if OOD is better on some classes
        "avg_acc_drop": drop_df["acc_drop"].mean(),

        # Mean positive drop: ignores classes where OOD improves
        "avg_positive_acc_drop": positive_drops.mean(),

        # Largest drop among classes
        "worst_acc_drop": worst_row["acc_drop"],
        "worst_drop_class": worst_row["label"],

        "num_drop_classes": len(drop_df),
    }

    return drop_metrics, drop_df


def compute_domain_metrics(
    results_df,
    in_domain_df,
    ood_df,
    labels,
    label_order=None,
):
    if label_order is None:
        label_order = labels

    splits = {
        "All": results_df,
        "In-domain": in_domain_df,
        "Out-of-domain": ood_df,
    }

    results = {}
    per_class_results = {}

    for split_name, split_df in splits.items():
        metrics, per_class = compute_metrics(
            split_df,
            labels=labels,
            label_order=label_order,
        )

        results[split_name] = metrics
        per_class_results[split_name] = per_class

    summary_df = pd.DataFrame(results).T

    drop_metrics, drop_df = compute_accuracy_drop(
        per_class_results["In-domain"],
        per_class_results["Out-of-domain"],
    )

    # Add drop metrics only to the Out-of-domain row
    for k, v in drop_metrics.items():
        summary_df.loc["Out-of-domain", k] = v

    summary_df = summary_df[
        [
            "num_samples",
            "num_present_classes",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "worst_class_accuracy",
            "worst_class",
            "avg_acc_drop",
            "avg_positive_acc_drop",
            "worst_acc_drop",
            "worst_drop_class",
            "num_drop_classes",
        ]
    ]

    return summary_df, per_class_results, drop_df

def compute_cl_domain_metrics(
    results_df,
    in_df,
    ood_df,
    labels,
    new_label,
    label_order=None,
    true_col="true_label",
    pred_col="pred_label",
):
    """
    Compute class-incremental metrics for All / In-domain / Out-of-domain.

    Metrics:
      - Old Bal. Acc.: balanced accuracy over old classes only
      - Old Macro-F1: macro-F1 over old classes only
      - New Acc.: accuracy/recall on true new-class samples
      - New F1: one-vs-rest F1 for the new class over the whole split
      - Old->New: percentage of true old-class samples predicted as the new class
    Assumes compute_metrics(...) returns percentage-valued metrics.
    """

    if label_order is None:
        label_order = labels

    old_labels = [lab for lab in label_order if lab != new_label]

    splits = {
        "All": results_df,
        "In-domain": in_df,
        "Out-of-domain": ood_df,
    }

    rows = {}
    old_per_class_results = {}

    for split_name, split_df in splits.items():
        split_df = split_df.copy()

        # -------------------------
        # Old-class metrics
        # -------------------------
        old_df = split_df[split_df[true_col].isin(old_labels)].copy()

        if len(old_df) > 0:
            old_metrics, old_per_class = compute_metrics(
                old_df,
                labels=old_labels,
                label_order=old_labels,
            )

            old_bal_acc = old_metrics["balanced_accuracy"]
            old_macro_f1 = old_metrics["macro_f1"]
            old_to_new = (old_df[pred_col] == new_label).mean() 
        else:
            old_metrics = {}
            old_per_class = pd.DataFrame()
            old_bal_acc = np.nan
            old_macro_f1 = np.nan
            old_to_new = np.nan

        # -------------------------
        # New-class accuracy
        # -------------------------
        new_true_df = split_df[split_df[true_col] == new_label].copy()

        if len(new_true_df) > 0:
            new_acc = (new_true_df[pred_col] == new_label).mean()
        else:
            new_acc = np.nan

        # -------------------------
        # New-class one-vs-rest F1
        # -------------------------
        if len(split_df) > 0 and (split_df[true_col] == new_label).any():
            y_true_new = (split_df[true_col] == new_label).astype(int)
            y_pred_new = (split_df[pred_col] == new_label).astype(int)

            new_f1 = f1_score(
                y_true_new,
                y_pred_new,
                zero_division=0,
            )
        else:
            new_f1 = np.nan

        rows[split_name] = {
            "num_samples": len(split_df),
            "num_old_samples": len(old_df),
            "num_new_samples": len(new_true_df),
            "old_balanced_accuracy": old_bal_acc,
            "old_macro_f1": old_macro_f1,
            "old_to_new": old_to_new,
            "new_accuracy": new_acc,
            "new_f1": new_f1
        }

        old_per_class_results[split_name] = old_per_class

    summary_df = pd.DataFrame(rows).T

    summary_df = summary_df[
        [
            "num_samples",
            "num_old_samples",
            "num_new_samples",
            "old_balanced_accuracy",
            "old_macro_f1",
            "old_to_new",
            "new_accuracy",
            "new_f1",
        ]
    ]

    return summary_df, old_per_class_results
