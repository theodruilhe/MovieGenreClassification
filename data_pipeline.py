# data/train_data.txt
# data/test_data.txt

import os
import string

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def load_data(text_file):
    """
    Load data from text file
    """
    data = []
    with open(text_file, "r") as f:
        for line in f:
            line = line.strip().split(":::")
            data.append(line)
    return pd.DataFrame(data, columns=["label", "title", "genre", "description"])


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
    # Create a CountVectorizer with optional max_features
    vectorizer = CountVectorizer(max_features=max_features)

    # Process the data in chunks
    chunks = [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame(index=df.index)

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")

        # Fit the vectorizer on the current chunk
        word_counts_sparse = vectorizer.fit_transform(
            chunk[col].apply(lambda x: " ".join(x))
        )

        # Convert the sparse matrix to a DataFrame
        word_count_df = pd.DataFrame(
            word_counts_sparse.toarray(),
            index=chunk.index,
            columns=vectorizer.get_feature_names_out(),
        )

        # Concatenate the results with the previous chunks
        result_df = pd.concat([result_df, word_count_df], axis=1)

    return pd.concat([df, result_df], axis=1)


if __name__ == "__main__":
    # Load data
    print("Loading data...")

    if not os.path.exists("data/train_data_embed.pkl"):
        train_data = load_data("data/train_data.txt")

        if not os.path.exists("data/train_data_wc.pkl"):
            if not os.path.exists("data/train_data_tok.pkl"):
                # Load spacy
                print("Loading spacy...")
                nlp = spacy.load("en_core_web_sm")
                tqdm.pandas()

                # Tokenize train_data if not already tokenized
                print("Tokenizing train_data...")
                train_data = tokenize_col(
                    train_data, ["description"], nlp, remove_stop=True
                )
                first_last_token(train_data)
                print("Saving tokenized train_data...")
                train_data.to_pickle("data/train_data_tok.pkl")
            else:
                train_data = pd.read_pickle("data/train_data_tok.pkl")

            # Create word count dataset in chunks for the sample data
            max_features = 200000  # Adjust as needed
            chunk_size = 1000  # Adjust based on available memory

            print("Creating word count dataset for the sample data...")
            train_data = create_word_count_dataset(
                train_data, max_features=max_features, chunk_size=chunk_size
            )

            # Save the word count dataset to a CSV file
            print("Saving word count dataset...")
            train_data.to_pickle("data/train_data_wc.pkl")
        else:
            train_data = pd.read_pickle("data/train_data_wc.pkl")

        # embedding from description tokens
        print("Embedding from tokens...")

        unique_tokens = train_data.description_t.explode().unique()
        print("Number of unique tokens: ", len(unique_tokens))

        if not os.path.exists("data/description_embedding.model"):
            model = Word2Vec(
                train_data.description_t,
                vector_size=100,
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
        print(f"Missing keys ({len(missing_keys)})")
        train_data.loc[:, "embedding"] = train_data.description_t.progress_apply(
            lambda x: np.nanmean(
                [model.wv[word] for word in set(x) & set(model.wv.index_to_key)], axis=0
            )
        )
        print("Saving data...")
        train_data.to_pickle("data/train_data_embed.pkl")
    else:
        train_data = pd.read_pickle("data/train_data_embed.pkl")
        print("Data loaded")
        model = Word2Vec.load("data/description_embedding.model")
        print("Word2Vec model loaded")

    """
    # PCA
    all_desc_vectors = []
    for desc in train_data.description_t:
        vec = np.stack([model.wv[word] for word in desc]).mean(axis=0)
        all_desc_vectors.append(vec)

    X = np.array(all_desc_vectors)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(X_normalized)

    print(X_reduced.shape)
    """
