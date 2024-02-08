import pandas as pd
pd.set_option('display.max_colwidth', None)

from utils import get_random_elements_from_cluster, create_clustered_col
from clustering import create_clustered_df, add_pca_features


if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df)

    full_df = create_clustered_df(pca_df, n_clusters=8)

    cluster_column = create_clustered_col(full_df, df)
    desc = get_random_elements_from_cluster(cluster_column, 'cluster', 0)
    print(desc)
