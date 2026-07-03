import ast
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_feature(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return np.asarray(x, dtype=np.float32)


def compute_tsne_df(
    results_df,
    label_col,
    max_samples_per_class=1000,
    seed=42,
    perplexity=30,
    exclude_labels=None,
):
    tsne_df = results_df.copy()

    if exclude_labels is not None:
        tsne_df = tsne_df[~tsne_df[label_col].isin(exclude_labels)]

    tsne_df = (
        tsne_df
        .groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max_samples_per_class), random_state=seed))
        .reset_index(drop=True)
    )

    if "true_accent" not in tsne_df.columns:
        tsne_df["true_accent"] = tsne_df[label_col]

    X = np.vstack(tsne_df["features"].apply(parse_feature).values)

    X = StandardScaler().fit_transform(X)

    n_pca = min(50, X.shape[1])
    X_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )

    Z = tsne.fit_transform(X_pca)

    tsne_df["tsne_1"] = Z[:, 0]
    tsne_df["tsne_2"] = Z[:, 1]

    return tsne_df