import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import filter_data_genre


def add_pca_features(df, n_components=37):
    embeddings = df.drop(
        [
            "label",
            "title",
            "genre",
            "description",
            "description_t",
            "first_token",
            "last_token",
            "embedding",
        ],
        axis=1,
    )
    labels = df["genre"]
    scaler = StandardScaler()
    scaler.fit(embeddings)
    embeddings = scaler.transform(embeddings)

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(embeddings)

    print(
        "Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_)[-1]
    )

    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=[f"pc_{i}" for i in range(1, n_components + 1)],
    )
    labels = labels.reset_index(drop=True)
    finalDf = pd.concat([principalDf, labels], axis=1)
    return finalDf, pca


def plot_pca(df):
    # plot the first two principal components
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)
    targets = df.genre.unique()
    colors = [
        "r",
        "g",
        "b",
        "c",
        "m",
        "y",
        "k",
        "orange",
        "purple",
        "brown",
    ]
    for target, color in zip(targets, colors):
        indicesToKeep = df["genre"] == target
        ax.scatter(
            df.loc[indicesToKeep, "pc_1"],
            df.loc[indicesToKeep, "pc_2"],
            c=color,
            s=6,
        )
    ax.legend(targets)
    ax.grid()
    plt.show()


def scree_plot(pca):
    # plot the explained variance
    plt.figure(figsize=(8, 8))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Scree Plot")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df, n_components=37)

    plot_pca(pca_df)
