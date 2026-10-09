import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    # General text
    "font.size": 12,

    # Axes
    "axes.titlesize": 15,
    "axes.labelsize": 14,

    # Legend
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,

    # Tick labels
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    # Export
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
})


def pretty_label(x):
    return str(x).replace("_", " ").replace("-", " ").title()


def plot_feat_tsne(
    ax,
    data,
    color_col,
    title,
    cmap_name="tab20",
    s=9,
    alpha=0.78,
    legend_title=None,
    show_counts=True,
    hide_ticks=True,
):
    df = data.dropna(subset=["tsne_1", "tsne_2", color_col]).copy()
    df["_label"] = df[color_col].astype(str)

    labels = sorted(df["_label"].unique())
    cmap = mpl.colormaps[cmap_name].resampled(len(labels))

    handles = []

    for i, label in enumerate(labels):
        g = df[df["_label"] == label]
        color = cmap(i)

        ax.scatter(
            g["tsne_1"],
            g["tsne_2"],
            s=s,
            alpha=alpha,
            color=color,
            edgecolors="none",
            linewidths=0,
            rasterized=True,   # keeps PDF size reasonable
        )

        legend_label = pretty_label(label)
        if show_counts:
            legend_label += f" ({len(g)})"

        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markersize=4.5,
                markerfacecolor=color,
                markeredgecolor="none",
                label=legend_label,
            )
        )

    ax.set_title(title, pad=6)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    ax.legend(
        handles=handles,
        title=legend_title,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.35,
    )

# def make_cm_df(df, labels):
#     cm = confusion_matrix(
#         df["true_label"],
#         df["pred_label"],
#         labels=labels,
#         normalize="true",
#     )

#     return pd.DataFrame(
#         cm,
#         index=[pretty_label(x) for x in labels],
#         columns=[pretty_label(x) for x in labels],
#     )


def make_cm_df(
    df,
    labels,
    true_col="true_label",
    pred_col="pred_label",
    outside_label="Outside",
    normalize="true",
):
    """
    Confusion matrix where rows are evaluated labels.
    Predictions outside labels are counted as errors and shown only if present.
    """

    df = df.copy()
    labels = [str(x) for x in labels]

    df[true_col] = df[true_col].astype(str)
    df[pred_col] = df[pred_col].astype(str)

    # Filter only by true labels.
    # Do not filter predictions.
    df = df[df[true_col].isin(labels)].copy()

    has_outside = (~df[pred_col].isin(labels)).any()

    if has_outside:
        df[pred_col] = df[pred_col].where(
            df[pred_col].isin(labels),
            outside_label,
        )
        pred_labels = labels + [outside_label]
    else:
        pred_labels = labels

    cm = pd.crosstab(
        df[true_col],
        df[pred_col],
        rownames=["True label"],
        colnames=["Predicted label"],
        dropna=False,
    )

    cm = cm.reindex(
        index=labels,
        columns=pred_labels,
        fill_value=0,
    )

    if normalize == "true":
        cm = cm.div(
            cm.sum(axis=1).replace(0, np.nan),
            axis=0,
        ).fillna(0)

    cm.index = [pretty_label(x) for x in cm.index]
    cm.columns = [pretty_label(x) for x in cm.columns]

    return cm
def plot_cm(ax, cm_df, title, show_ylabel=True, show_cbar=False, cbar_ax=None):
    def _make_annot_df(cm_df):
        try:
            return cm_df.map(lambda x: f"{x:.2f}" if x > 0 else "")
        except Exception as e:
            return cm_df.applymap(lambda x: f"{x:.2f}" if x > 0 else "")
    sns.heatmap(
        cm_df,
        ax=ax,
        annot=_make_annot_df(cm_df),
        fmt="",
        cmap="turbo",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar=show_cbar,
        cbar_ax=cbar_ax,
        annot_kws={"size": 7},
    )

    ax.set_title(title, pad=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label" if show_ylabel else "")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    if not show_ylabel:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)