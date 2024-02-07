import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import seaborn as sns

from pca_embed import add_pca_features
from utils import filter_data_genre


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


def create_clustered_df(pca_df, n_clusters=8):

    pca_mat = pca_df.drop(columns=["genre"])
    kmeans = KMeans(
        n_clusters=n_clusters, init="random", max_iter=300, n_init=10, random_state=29
    )
    kmeans.fit(pca_mat)
    pca_df["cluster"] = kmeans.labels_

    return pca_df


def visualize_clusters(df):
    """
    Plot the data colored by cluster on the first two principal components, with a
    legend.
    """
    plt.figure(figsize=(10, 10))
    colors = ["red", "green", "blue", "purple", "orange", "yellow", "pink", "black"]
    for color, cluster in zip(colors, df["cluster"].unique()):
        plt.scatter(
            df[df["cluster"] == cluster]["pc_1"],
            df[df["cluster"] == cluster]["pc_2"],
            label=cluster,
            s=8,
            c=color,
        )
    plt.legend()
    plt.show()





def cluster_analysis(df):

    pca_df = add_pca_features(df)

    pca_mat = pca_df.drop(columns=["genre"])


    kmeans = KMeans(
    n_clusters=8, init="random", max_iter=300, n_init=10, random_state=29
    )
    kmeans.fit(pca_mat)
    pca_df["cluster"] = kmeans.labels_

    genre_cluster_distribution = pca_df.groupby(['cluster', 'genre']).size().reset_index(name='count')
    
    pivot_table = pd.pivot_table(genre_cluster_distribution, values='count', index='cluster', columns='genre', fill_value=0)

    percentage_table = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100

    print(round(percentage_table,1))

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle("Movie genre distribution in each cluster", fontsize=16)


    plt.subplots_adjust(wspace=0.5)

    for i, ax in enumerate(axes.flatten()):

        cluster_data = genre_cluster_distribution[genre_cluster_distribution['cluster'] == i]
        
        ax.pie(cluster_data['count'], labels=cluster_data['genre'], autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Cluster {i}")

    plt.show()



if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df)

    full_df = create_clustered_df(pca_df, n_clusters=8)

    visualize_clusters(full_df)

    cluster_analysis(filtered_df)

