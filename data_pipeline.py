# data/train_data.txt
# data/test_data.txt

import os

import pandas as pd
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

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


def create_word_count_dataset(df, col="description_t", max_features=None, chunk_size=1000):
    # Create a CountVectorizer with optional max_features
    vectorizer = CountVectorizer(max_features=max_features)

    # Process the data in chunks
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Initialize an empty DataFrame to store the results
    result_df = pd.DataFrame(index=df.index)

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...")
        
        # Fit the vectorizer on the current chunk
        word_counts_sparse = vectorizer.fit_transform(chunk[col].apply(lambda x: ' '.join(x)))

        # Convert the sparse matrix to a DataFrame
        word_count_df = pd.DataFrame(word_counts_sparse.toarray(), index=chunk.index, columns=vectorizer.get_feature_names_out())

        # Concatenate the results with the previous chunks
        result_df = pd.concat([result_df, word_count_df], axis=1)

    return pd.concat([df, result_df], axis=1)

# ... (existing code)

if __name__ == "__main__":
    # Load data
    print("Loading data...")
    train_data = load_data("data/train_data.txt")

    # Load spacy
    print("Loading spacy...")
    nlp = spacy.load("en_core_web_sm")
    tqdm.pandas()

    # Tokenize train_data if not already tokenized
    print("Tokenizing train_data...")
    if not os.path.exists("data/train_data.pkl"):
        train_data = tokenize_col(train_data, ["description"], nlp, remove_stop=True)
        print("Saving tokenized train_data...")
        train_data.to_pickle("data/train_data.pkl")
    else:
        train_data = pd.read_pickle("data/train_data.pkl")

    print(train_data.head())

    first_last_token(train_data)
    print("\n5 most common first tokens")
    print(train_data.first_token.value_counts().head(5))

    print("5 most common last tokens")
    print(train_data.last_token.value_counts().head(5))

    # Load a sample of the data
    sample_data = load_data("data/train_data.txt").head(50)

    # Tokenize sample_data if not already tokenized
    print("Tokenizing sample_data...")
    if not os.path.exists("data/sample_data.pkl"):
        sample_data = tokenize_col(sample_data, ["description"], nlp, remove_stop=True)
        print("Saving tokenized sample_data...")
        sample_data.to_pickle("data/sample_data.pkl")
    else:
        sample_data = pd.read_pickle("data/sample_data.pkl")

    # Create word count dataset in chunks for the sample data
    max_features = 200000  # Adjust as needed
    chunk_size = 1000   # Adjust based on available memory

    print("Creating word count dataset for the sample data...")
    word_count_data = create_word_count_dataset(sample_data, max_features=max_features, chunk_size=chunk_size)

    # Display the resulting dataset
    print(word_count_data.head())

    # Save the word count dataset to a CSV file
    print("Saving word count dataset...")
    word_count_data.to_csv("data/word_count_data.csv", index=False)
