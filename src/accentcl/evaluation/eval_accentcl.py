import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


@torch.no_grad()
def evaluate(
    model,
    loader,
    old_label2id,
    label2id,
    device,
    save_predictions_path=None
):
    """
    Evaluate class-incremental accent classifier.

    Assumes:
      - old classes keep the same ids as the base model
      - new class is appended as the last class

    Returns:
      overall expanded-set metrics
      old-class metrics
      new-class metrics
      per-class metrics
      confusion matrix
      optional prediction dataframe
    """
    model.eval()

    assert (
        len(label2id) == len(old_label2id) + 1
        and all(label in label2id and label2id[label] == old_id for label, old_id in old_label2id.items())
        and max(label2id.values()) == len(label2id) - 1
    ), "label2id must equal old_label2id plus exactly one new class appended at the last index."

    all_true = []
    all_pred = []
    rows = []
    id2label = {v: k for k, v in label2id.items()}
    for batch_data in tqdm(loader, desc="Evaluating continual", leave=False):
        wavs = batch_data["wavs"].to(device)
        wav_lens = batch_data["wav_lens"].to(device)
        wav_lens = (wav_lens * wavs.shape[1]).long()
        labels = batch_data["labels"].to(device)
        audio_paths = batch_data["audio_paths"]
        if "in_domain" in batch_data:
            in_domain = batch_data["in_domain"]

        out = model(wavs, length=wav_lens)
        logits = out["accent_logits"]
        features = out["features"]
        probs = torch.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)

        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        if save_predictions_path is not None:
            for i in range(len(preds)):
                true_id = int(labels[i].cpu().item())
                pred_id = int(preds[i].cpu().item())

                rows.append({
                    "audio_path": audio_paths[i],
                    "true_label_id": true_id,
                    "pred_label_id": pred_id,
                    "true_label": id2label[true_id],
                    "pred_label": id2label[pred_id],
                    "confidence": float(conf[i].cpu().item()),
                    "true_source": audio_paths[i].split("/")[-5],
                    "features": features[i].cpu().numpy().tolist(),
                    "in_domain": bool(in_domain[i].cpu().item()) if "in_domain" in batch_data else None,
                })


    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    new_class_id = len(old_label2id)
    old_mask = y_true < new_class_id
    new_mask = y_true == new_class_id

    # -------------------------
    # Expanded (all) class metrics
    # -------------------------
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    class_ids = list(range(len(label2id)))
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )


    # -------------------------
    # Per-class metrics
    # -------------------------
    per_class_acc = {}
    for cls_id in class_ids:
        cls_name = id2label[cls_id]

        cls_mask = y_true == cls_id

        if cls_mask.any():
            cls_acc = float((y_pred[cls_mask] == cls_id).mean())
        else:
            cls_acc = 0.0

        per_class_acc[cls_name] = cls_acc
    worst_class_acc = min(per_class_acc.values()) if per_class_acc else 0.0

    # -------------------------
    # Old-class metrics
    # -------------------------
    old_class_ids = list(range(len(old_label2id)))
    if old_mask.any():
        _, old_recall, old_f1, _ = precision_recall_fscore_support(
            y_true[old_mask],
            y_pred[old_mask],
            labels=old_class_ids,
            zero_division=0,
        )
        old_bal_acc = float(old_recall.mean())
        old_macro_f1 = float(old_f1.mean())
    else:
        old_bal_acc = 0.0
        old_macro_f1 = 0.0


    per_old_accuracy = {}
    for cls_id in old_class_ids:
        cls_name = id2label[cls_id]
        cls_mask = y_true == cls_id

        if cls_mask.any():
            cls_acc = float((y_pred[cls_mask] == cls_id).mean())
        else:
            cls_acc = 0.0

        per_old_accuracy[cls_name] = cls_acc

    worst_old_accuracy = (
        min(per_old_accuracy.values()) if per_old_accuracy else 0.0
    )

    # -------------------------
    # New-class metrics
    # -------------------------
    if new_mask.any():
        new_accuracy = float((y_pred[new_mask] == new_class_id).mean())

        new_precision, new_recall, new_f1, new_support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[new_class_id],
            zero_division=0,
        )

        new_precision = float(new_precision[0])
        new_recall = float(new_recall[0])
        new_f1 = float(new_f1[0])
        new_support = int(new_support[0])
    else:
        new_accuracy = 0.0
        new_precision = 0.0
        new_recall = 0.0
        new_f1 = 0.0
        new_support = 0

    harmonic_old_new_f1 = (
        2 * old_macro_f1 * new_f1 / (old_macro_f1 + new_f1 + 1e-8)
    )

    results = {
        # Expanded all-class metrics
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "worst_class_accuracy": float(worst_class_acc),

        # Old-class retention metrics
        "old_balanced_accuracy": float(old_bal_acc),
        "old_macro_f1": float(old_macro_f1),
        "worst_old_accuracy": float(worst_old_accuracy),

        # New-class learning metrics
        "new_accuracy": float(new_accuracy),
        "new_precision": float(new_precision),
        "new_recall": float(new_recall),
        "new_f1": float(new_f1),
        "new_support": int(new_support),

        # Stability-plasticity summary
        "harmonic_mean_old_new_f1": float(harmonic_old_new_f1),
    }

    if save_predictions_path is not None:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(save_predictions_path, index=False)

    return results