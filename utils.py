"""
This file contains utility functions for data processing.
"""
import pandas as pd


def filter_data_genre(df, minimum_threshold=1000):
    genre_freqs = df.genre.value_counts()
    selected_genres = genre_freqs >= minimum_threshold
    return df.loc[df.genre.isin(selected_genres[selected_genres].index)]

def create_clustered_col(df_cluster, df):
    new_df =  pd.concat([df_cluster['cluster'], df], axis=1)
    return new_df

def get_random_elements_from_cluster(df, cluster_column, cluster_value):

    cluster_data = df[df[cluster_column] == cluster_value]
    random_elements = cluster_data.sample(n=2)

    return random_elements['description']




