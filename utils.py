"""
This file contains utility functions for data processing.
"""


def filter_data_genre(df, minimum_threshold=1000):
    genre_freqs = df.genre.value_counts()
    selected_genres = genre_freqs >= minimum_threshold
    return df.loc[df.genre.isin(selected_genres[selected_genres].index)]
