import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

@torch.no_grad()
def evaluate(
    model,
    loader,
    label2id,
    device,
    save_predictions_path=None
):
    """
    Evaluate a fixed-label base accent classifier.

    Works with models that return:
        {"accent_logits": logits, "features": ..., ...}

    Returns:
        overall accuracy
        balanced accuracy
        macro-F1
        worst-class accuracy
        optional prediction dataframe
    """
    model.eval()
    id2label = {v: k for k, v in label2id.items()}

    all_true = []
    all_pred = []
    rows = []

    for batch_data in tqdm(loader, desc="Evaluating base", leave=False):
        wavs = batch_data["wavs"].to(device)
        wav_lens = batch_data["wav_lens"].to(device)
        wav_lens = (wav_lens * wavs.shape[1]).long() # convert back to absolute lengths
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
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    class_ids = list(range(len(label2id)))
    macro_f1 = f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)

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

    results = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "worst_class_accuracy": float(worst_class_acc),
    }

    if save_predictions_path is not None:
        results_df = pd.DataFrame(rows)
        results_df.to_csv(save_predictions_path, index=False)

    return results