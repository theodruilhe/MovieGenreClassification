import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def add_pca_features(df, n_components=39):
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
    print("Embeddings shape:", embeddings.shape)
    labels = df["genre"]
    print("Labels shape:", labels.shape)
    scaler = StandardScaler()
    scaler.fit(embeddings)
    embeddings = scaler.transform(embeddings)
    print("Embeddings shape after scaling:", embeddings.shape)

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(embeddings)
    print("Principal components shape:", principalComponents.shape)

    print(
        "Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_)[-1]
    )

    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=[f"pc_{i}" for i in range(1, n_components + 1)],
    )
    print("Principal components df shape:", principalDf.shape)
    labels = labels.reset_index(drop=True)
    finalDf = pd.concat([principalDf, labels], axis=1)
    print("Final df shape:", finalDf.shape)
    return finalDf


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


if __name__ == "__main__":
    df = pd.read_csv("data/train_data_embed.csv")
    genre_freqs = df.genre.value_counts()

    minimum_threshold = 1000
    selected_genres = genre_freqs >= minimum_threshold

    filtered_df = df.loc[df.genre.isin(selected_genres[selected_genres].index)]

    pca_df = add_pca_features(filtered_df)

    plot_pca(pca_df)
