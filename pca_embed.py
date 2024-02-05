import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    df = pd.read_csv("data/train_data_embed.csv")
    genre_freqs = df.genre.value_counts()

    minimum_threshold = 1000
    selected_genres = genre_freqs >= minimum_threshold

    filtered_df = df.loc[df.genre.isin(selected_genres[selected_genres].index)]

    embeddings = filtered_df.drop(
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
    labels = filtered_df["genre"]

    scaler = StandardScaler()
    scaler.fit(embeddings)
    embeddings = scaler.transform(embeddings)

    pca = PCA(n_components=80)

    principalComponents = pca.fit_transform(embeddings)

    # sree plot to choose the number of components
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.show()

    # 37 components explain 80% of the variance
    pca = PCA(n_components=37)
    principalComponents = pca.fit_transform(embeddings)

    principalDf = pd.DataFrame(
        data=principalComponents, columns=[f"pc_{i}" for i in range(1, 38)]
    )
    finalDf = pd.concat([principalDf, labels], axis=1)

    targets = selected_genres[selected_genres].index
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
    # plot in 3D interactive
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_zlabel("Principal Component 3", fontsize=15)
    ax.set_title("3 component PCA", fontsize=20)
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf["genre"] == target
        ax.scatter(
            finalDf.loc[indicesToKeep, "pc_1"],
            finalDf.loc[indicesToKeep, "pc_2"],
            finalDf.loc[indicesToKeep, "pc_3"],
            c=color,
            s=50,
        )
    ax.legend(targets)
    ax.grid()
    plt.show()
