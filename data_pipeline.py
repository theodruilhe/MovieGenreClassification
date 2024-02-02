# data/train_data.txt
# data/test_data.txt

import os

import pandas as pd
import spacy
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


def first_last_token(df, col="description_t"):
    df.loc[:, "first_token"] = df[col].apply(lambda x: x[0])
    df.loc[:, "last_token"] = df[col].apply(lambda x: x[-1])


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_data = load_data("data/train_data.txt")

    # Load spacy
    print("Loading spacy...")
    nlp = spacy.load("en_core_web_sm")
    tqdm.pandas()
    # Tokenize
    print("Tokenizing...")
    if not os.path.exists("data/train_data.pkl"):
        train_data = tokenize_col(train_data, ["description"], nlp, remove_stop=True)
        print("Saving data...")
        train_data.to_pickle("data/train_data.pkl")
    else:
        train_data = pd.read_pickle("data/train_data.pkl")

    print(train_data.head())

    first_last_token(train_data)
    print("\n5 most common first tokens")
    print(train_data.first_token.value_counts().head(5))

    print("5 most common last tokens")
    print(train_data.last_token.value_counts().head(5))
