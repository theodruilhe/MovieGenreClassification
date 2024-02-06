import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

from pca_embed import add_pca_features


def elbow_method(data):
    wcss = []
    for i in tqdm(range(1, 11)):
        kmeans = KMeans(
            n_clusters=i, init="random", max_iter=300, n_init=10, random_state=29
        )
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/train_data_embed.csv")
    genre_freqs = df.genre.value_counts()

    minimum_threshold = 1000
    selected_genres = genre_freqs >= minimum_threshold

    filtered_df = df.loc[df.genre.isin(selected_genres[selected_genres].index)]

    pca_df = add_pca_features(filtered_df)

    pca_mat = pca_df.drop(columns=["genre"])
    # elbow_method(pca_mat)

    kmeans = KMeans(
        n_clusters=8, init="random", max_iter=300, n_init=10, random_state=29
    )
    kmeans.fit(pca_mat)
    pca_df["cluster"] = kmeans.labels_

    # check the distribution of clusters
    print(pca_df.cluster.value_counts())

    # visualize the clusters
    plt.scatter(
        pca_df["pc_1"], pca_df["pc_2"], c=pca_df["cluster"], s=8, cmap="viridis"
    )
    plt.title("Clusters")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.show()
