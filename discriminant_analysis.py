import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pca_embed import add_pca_features
from utils import filter_data_genre


def discriminant_analysis_pca(pca_df, test_size=0.2, random_state=42, heatmap=False):
    X_original = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y, test_size=test_size, random_state=random_state
    )

    # Reset indexes to align correctly after splitting
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Initialize and train Linear Discriminant Analysis model
    lda = LinearDiscriminantAnalysis(n_components=4)
    lda.fit(X_train, y_train)

    transformed_data_train = lda.transform(X_train)
    transformed_data_test = lda.transform(X_test)

    df_train_da = pd.DataFrame(
        transformed_data_train, columns=[f"da_{i}" for i in range(4)]
    )
    df_test_da = pd.DataFrame(
        transformed_data_test, columns=[f"da_{i}" for i in range(4)]
    )

    # Predict using the original feature space
    y_pred = lda.predict(X_test)
    y_pred_train = lda.predict(X_train)

    # Concatenate transformed data with true and predicted genres, ensuring alignment
    full_transformed_data_train = pd.concat(
        [
            df_train_da,
            y_train.reset_index(drop=True),
            pd.Series(y_pred_train, name="pred"),
        ],
        axis=1,
    )
    full_transformed_data_test = pd.concat(
        [df_test_da, y_test.reset_index(drop=True), pd.Series(y_pred, name="pred")],
        axis=1,
    )

    # Print classification report
    print("Report on PCA data:")
    print(classification_report(y_test, y_pred))
    if heatmap:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Heatmap visualization
        plt.figure(figsize=(10, 6))
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Blues")
        plt.title("Classification Report Heatmap")
        plt.show()

    return lda, full_transformed_data_train, full_transformed_data_test


def plot_lda(pca_df, lda):
    X = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

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


def show_explained_variance(lda):
    for i, explained_variance in enumerate(lda.explained_variance_ratio_):
        print(f"Explained variance of DA_{i}: {explained_variance:.3f}")
    print(f"Total explained variance: {lda.explained_variance_ratio_.sum():.3f}")


if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df, n_components=37)

    # discriminant_analysis(filtered_df)
    lda, train_da, test_da = discriminant_analysis_pca(pca_df, heatmap=False)
    show_explained_variance(lda)
