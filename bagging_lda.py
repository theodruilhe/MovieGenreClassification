import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tqdm import tqdm

from pca_embed import add_pca_features


def bagging_discriminant_analysis(
    pca_df, n_estimators=10, test_size=0.2, random_state=42
):

    pca_df_train, pca_df_test = train_test_split(
        pca_df, test_size=test_size, random_state=random_state
    )

    X_test = pca_df_test.drop("genre", axis=1)
    y_test = pca_df_test["genre"]

    test_preds = []

    # Initialize and train Linear Discriminant Analysis model

    for _ in tqdm(range(n_estimators), desc="Bagging progress"):
        # Generate a bootstrap sample
        bootstrap_sample = resample(pca_df_train, replace=True)

        X_train = bootstrap_sample.drop("genre", axis=1)
        y_train = bootstrap_sample["genre"]

        lda = LinearDiscriminantAnalysis(n_components=4)

        lda.fit(X_train, y_train)

        # Predictions on the test set
        y_pred_test = lda.predict(X_test)

        # Store predictions
        test_preds.append(y_pred_test)

    full_transformed_data_test = pd.DataFrame(test_preds).T.add_prefix("pred_da_")
    full_transformed_data_test = pd.concat(
        [y_test.reset_index(drop=True), full_transformed_data_test], axis=1
    )

    # Find the most common prediction across all LDAs
    full_transformed_data_test["pred_maj_vote"] = full_transformed_data_test.iloc[
        :, 1:
    ].mode(axis=1)[0]

    # Rename the second column
    full_transformed_data_test.columns = ["genre", "pred_maj_vote"] + [
        f"pred_da_{i}" for i in range(1, n_estimators + 1)
    ]

    print(full_transformed_data_test.head(30))

    return full_transformed_data_test


def evaluate_predictions(y_true, y_pred):
    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    # Load data and perform PCA
    df = pd.read_csv("data/full_data_embed.csv")
    pca_df, _ = add_pca_features(df, n_components=37)

    # Perform bagging on LDA with PCA
    test_preds = bagging_discriminant_analysis(
        pca_df, n_estimators=100, test_size=0.2, random_state=42
    )

    # Extract true genre labels and majority vote predictions
    y_true = test_preds["genre"]
    y_pred = test_preds["pred_maj_vote"]

    # Evaluate predictions
    evaluate_predictions(y_true, y_pred)
