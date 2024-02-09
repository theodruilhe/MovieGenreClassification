"""
This file contains utility functions for data processing.
"""

import pandas as pd


def filter_data_genre(df, minimum_threshold=1000):
    genre_freqs = df.genre.value_counts()
    selected_genres = genre_freqs >= minimum_threshold
    return df.loc[df.genre.isin(selected_genres[selected_genres].index)]


def create_clustered_col(df_cluster, df):
    """
    Create a new DataFrame with the cluster column
    Args:
      :df_cluster: DataFrame. The DataFrame containing the cluster column
      :df: DataFrame. The original DataFrame
    Returns:
      DataFrame. The DataFrame containing the cluster column
    """
    return pd.concat([df_cluster["cluster"], df], axis=1)


def get_random_elements_from_cluster(
    df, cluster_value, n_sample, cluster_column="cluster", random_state=29
):
    """
    Get random elements from a cluster
    Args:
      :df: DataFrame. The DataFrame containing the cluster column
      :cluster_value: int. The id of the cluster
      :n_sample: int. The number of random elements to get
      :cluster_column (Optional): str. The name of the cluster column
      :random_state (Optional): int. The random state to use
    Returns:
      DataFrame. The DataFrame containing the random elements
    """

    cluster_data = df[df[cluster_column] == cluster_value]
    random_elements = cluster_data.sample(n=n_sample, random_state=random_state)

    return random_elements
