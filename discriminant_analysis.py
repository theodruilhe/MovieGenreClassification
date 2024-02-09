import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

    # Predict using the original feature space
    y_pred = lda.predict(X_test)

    # Print classification report
    print("Report on PCA data:")
    print(classification_report(y_test, y_pred))

    return lda, classification_report(y_test, y_pred)


def plot_lda(pca_df, lda):
    X = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    lda = LinearDiscriminantAnalysis(n_components=2)
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

    # discriminant_analysis(filtered_df)
    lda, report_test = discriminant_analysis_pca(pca_df)
    plot_lda(pca_df, lda)
