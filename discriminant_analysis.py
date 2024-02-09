import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv

from pca_embed import add_pca_features
from utils import filter_data_genre


def discriminant_analysis_pca(pca_df, test_size=0.2, random_state=42):
    X_original = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train Linear Discriminant Analysis model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    transformed_data_train = lda.transform(X_train)
    transformed_data_test = lda.transform(X_test)

    df_train_da = pd.DataFrame(
        transformed_data_train, columns=[f"da_{i}" for i in range(9)]
    )
    df_test_da = pd.DataFrame(
        transformed_data_test, columns=[f"da_{i}" for i in range(9)]
    )

    # Predict using the original feature space
    y_pred = lda.predict(X_test)
    y_pred_train = lda.predict(X_train)

    full_transformed_data_train = pd.concat(
        [
            df_train_da,
            pd.DataFrame(y_train),
            pd.DataFrame(y_pred_train, columns=["pred"]),
        ],
        axis=1,
    )
    full_transformed_data_test = pd.concat(
        [df_test_da, pd.DataFrame(y_test), pd.DataFrame(y_pred, columns=["pred"])],
        axis=1,
    )

    # Print classification report
    print("Report on PCA data:")
    print(classification_report(y_test, y_pred))

    return lda, full_transformed_data_train, full_transformed_data_test


def plot_lda(pca_df, lda):
    X = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    lda = LinearDiscriminantAnalysis()
    X_r2 = lda.fit_transform(X, y)

    plt.figure(figsize=(8, 6))
    colors = [
        "red",
        "green",
        "blue",
        "purple",
        "orange",
        "yellow",
        "black",
        "pink",
        "brown",
        "gray",
    ]
    for color, target in zip(colors, np.unique(y)):
        plt.scatter(
            X_r2[y == target][:, 0],
            X_r2[y == target][:, 1],
            c=color,
            s=6,
            label=target,
        )

    plt.legend(title="Genre", loc="best", fontsize="small", title_fontsize="small")
    plt.title("Transformed Data (Linear Discriminant Analysis)", fontsize=16)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df, n_components=37)
    X_original = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    csv_train_path = "data/pca_df_train.csv"
    csv_test_path = "data/pca_df_test.csv"

    # Writing to CSV file
    train_df.to_csv(csv_train_path, index=False)
    test_df.to_csv(csv_test_path, index=False)
