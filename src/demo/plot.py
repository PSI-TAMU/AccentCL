import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion(
    results,
    label2ids,
    normalize=True,
    show_only_present=True,
    model_names=None,
    cmap="turbo",
    figsize=None,
):
    """
    results:
        dict[model_name][true_label] -> list[pred_label]

    label2ids:
        dict[model_name] -> dict[label] -> id
    """

    pretty_names = {
        "north_american": "North\nAmerican",
        "british_isles": "British\nIsles",
        "australasian": "Australasian",
        "south_asian": "South\nAsian",
        "southeast_asian": "Southeast\nAsian",
        "spanish": "Spanish",
        "chinese": "Chinese",
    }

    n_models = len(results)

    if figsize is None:
        figsize = (7.0 * n_models, 5.5)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    outputs = {}
    last_im = None

    for ax, (model_key, pred_dict) in zip(axes, results.items()):
        label2id = label2ids[model_key]

        all_labels = [k for k, _ in sorted(label2id.items(), key=lambda x: x[1])]

        if show_only_present:
            used = set(pred_dict.keys())
            for preds in pred_dict.values():
                used.update(preds)
            labels = [x for x in all_labels if x in used]
        else:
            labels = all_labels

        cm = pd.DataFrame(0, index=labels, columns=labels, dtype=float)

        for true_label, preds in pred_dict.items():
            if true_label not in labels:
                continue
            for pred_label in preds:
                if pred_label not in labels:
                    continue
                cm.loc[true_label, pred_label] += 1

        row_sums = cm.sum(axis=1).replace(0, np.nan)

        if normalize:
            cm_plot = cm.div(row_sums, axis=0) * 100
            vmin, vmax = 0, 100
            cbar_label = "Percentage (%)"
        else:
            cm_plot = cm.copy()
            vmin, vmax = 0, None
            cbar_label = "Count"

        correct = np.trace(cm.values)
        total = cm.values.sum()
        acc = correct / total * 100 if total > 0 else 0

        tick_labels = [
            pretty_names.get(x, x.replace("_", "\n").title())
            for x in labels
        ]

        last_im = ax.imshow(
            cm_plot.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticklabels(tick_labels, fontsize=9)

        plt.setp(
            ax.get_xticklabels(),
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )

        # white cell grid
        ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # cell text
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = cm_plot.iloc[i, j]
                count = int(cm.iloc[i, j])
                total_row = int(row_sums.iloc[i]) if not np.isnan(row_sums.iloc[i]) else 0

                if count == 0 or np.isnan(value):
                    continue

                if normalize:
                    text = f"{value:.0f}%\n{count}/{total_row}"
                else:
                    text = str(count)

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 55 else "black",
                    fontweight="bold" if i == j else "normal",
                )

        display_name = model_names.get(model_key, model_key) if model_names else model_key

        ax.set_title(
            f"{display_name}\nAcc. = {acc:.1f}%",
            fontsize=12,
            pad=10,
        )

        ax.set_xlabel("Predicted Accent", fontsize=10)
        ax.set_ylabel("True Accent", fontsize=10)

        for spine in ax.spines.values():
            spine.set_visible(False)

        outputs[model_key] = {
            "cm_counts": cm,
            "cm_plot": cm_plot,
            "acc": acc,
        }

    # one shared colorbar
    cbar = fig.colorbar(
        last_im,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label(cbar_label)

    plt.show()

    return outputs