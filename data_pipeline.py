# data/train_data.txt
# data/test_data.txt

import math
import multiprocessing
import os
import string
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def load_data(text_file):
    """
    Load data from text file
    """
    data = []
    with open(text_file, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip().split(":::")
            data.append(line)
    return pd.DataFrame(data, columns=["label", "title", "genre", "description"])


def merge_train_test(train_data_file, test_data_file, open_file):
    """
    Create a single dataframe from train and test data in the same format of the
    original data
    """
    train_data = load_data(train_data_file)
    test_data = load_data(test_data_file)
    merged = pd.concat([train_data, test_data], axis=0)
    with open(open_file, "w", encoding="utf8") as f:
        for i, row in tqdm(merged.iterrows()):
            f.write(":::".join(row) + "\n")


def regroup_genres(data):
    """
    Create new genres by grouping the existing ones
    """
    new_genres = {
        "thriller": "thriller/horror",
        "horror": "thriller/horror",
        "war": "action",
        "action": "action",
        "adventure": "action",
        "sci-fi": "action",
        "western": "action",
        "drama": "drama",
        "romance": "drama",
        "comedy": "comedy",
        "family": "family",
        "animation": "family",
        "music": "music",
        "musical": "music",
        "documentary": "documentary",
        "biography": "documentary",
        "history": "documentary",
        "game-show": "live",
        "sport": "live",
        "reality-tv": "live",
        "news": "live",
        "talk-show": "live",
        "mystery": "police",
        "fantasy": "police",
        "crime": "police",
        "adult": "adult",
        "comedy": "comedy",
        "short": "short",
    }
    data["genre"] = data["genre"].str.strip()

    data["genre"] = data["genre"].map(new_genres)
    data = data[data["genre"] != "adult"]

    return data


def tokenizer_lang(text, nlp, remove_stop=False):
    doc = nlp.tokenizer(text)

    if remove_stop:
        tokens = [t.text.lower() for t in doc if not t.is_punct and not t.is_stop]
    else:
        tokens = [t.text.lower() for t in doc if not t.is_punct]

    # remove blank space from tokens
    tokens = [t for t in tokens if t.strip()]

    return tokens


def tokenize_col(df, cols, nlp, remove_stop=False):
    for col in cols:
        df.loc[:, col + "_t"] = df.progress_apply(
            lambda x: tokenizer_lang(x[col], nlp, remove_stop), axis=1
        )

    return df


def extra_preprocessing(text):
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator).lower()
    return text


def first_last_token(df, col="description_t"):
    df.loc[:, "first_token"] = df[col].str[0].apply(lambda x: extra_preprocessing(x))
    df.loc[:, "last_token"] = df[col].str[-1].apply(lambda x: extra_preprocessing(x))


def create_word_count_dataset(
    df, col="description_t", max_features=None, chunk_size=1000
):
    vectorizer = CountVectorizer(max_features=max_features)

    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    result_df = pd.DataFrame(index=df.index)

    with tqdm(total=len(chunks)) as pbar:
        for i, chunk in enumerate(chunks):
            pbar.set_description(f"Processing chunk {i + 1}/{len(chunks)}...")

            word_counts_sparse = vectorizer.fit_transform(
                chunk[col].apply(lambda x: " ".join(x))
            )

            word_count_df = pd.DataFrame(
                word_counts_sparse.toarray(),
                index=chunk.index,
                columns=vectorizer.get_feature_names_out(),
            )

            result_df = pd.concat([result_df, word_count_df], axis=1)
            pbar.update(1)

    return pd.concat([df, result_df], axis=1)


def main(filename, save_file=True):
    print("Loading data...")
    train_data = load_data(filename)

    print("Loading spacy...")
    nlp = spacy.load("en_core_web_sm")
    tqdm.pandas()

    print("Regrouping genres...")
    train_data = regroup_genres(train_data)

    print("Tokenizing data...")
    train_data = tokenize_col(train_data, ["description"], nlp, remove_stop=True)
    first_last_token(train_data)

    print("Embedding from tokens...")
    unique_tokens = train_data.description_t.explode().unique()
    print("Number of unique tokens: ", len(unique_tokens))
    vector_size = 100

    if not os.path.exists("data/description_embedding.model"):
        model = Word2Vec(
            train_data.description_t,
            vector_size=vector_size,
            window=5,
            min_count=1,
            sg=0,
            epochs=30,
            workers=4,
        )
        model.save("data/description_embedding.model")
        print("Word2Vec model saved")
    else:
        model = Word2Vec.load("data/description_embedding.model")
        print("Word2Vec model loaded")

    print("Adding embeddings to dataframe...")
    unique_descriptions = set(
        [item for sublist in train_data.description_t for item in sublist]
    )
    missing_keys = list(set(unique_descriptions) - set(model.wv.index_to_key))
    print(f"Missing keys: {len(missing_keys)}")
    train_data.loc[:, "embedding"] = train_data.description_t.progress_apply(
        lambda x: np.nanmean(
            [model.wv[word] for word in set(x) & set(model.wv.index_to_key)], axis=0
        )
    )

    for i in range(vector_size):
        train_data.loc[:, f"embedding_{i}"] = train_data.embedding.apply(
            lambda x: x[i] if not np.isnan(x[i]) else 0
        )

    if save_file:
        print("Saving data...")
        output_filename = filename.split(".")[0] + "_embed.csv"
        train_data.to_csv(output_filename, index=False)

    return train_data


if __name__ == "__main__":
    print("Merging train and test data...")
    merge_train_test(
        "data/train_data.txt", "data/test_data_solution.txt", "data/full_data.txt"
    )
    main("data/full_data.txt")
