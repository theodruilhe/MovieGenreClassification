import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pca_embed import add_pca_features
from utils import filter_data_genre


def discriminant_analysis(df):
    labels = df["genre"]
    labels = labels.reset_index(drop=True)
    
    df = pd.concat([df.iloc[:, 8:], labels], axis=1)
    df.dropna(inplace=True)

    X = df.drop("genre", axis=1)
    y = df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Linear Discriminant Analysis model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict
    y_pred = lda.predict(X_test)

    # Print classification report
    print("Report on Embeddings data:")
    print(classification_report(y_test, y_pred))


def discriminant_analysis_pca(df):
    pca_df = add_pca_features(df, n_components=99)

    X= pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Linear Discriminant Analysis model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict
    y_pred = lda.predict(X_test)

    # Print classification report
    print("Report on PCA data:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    df = pd.read_csv("data/train_data_embed.csv")
   
    filtered_df = filter_data_genre(df, minimum_threshold=1000)
    
    discriminant_analysis_pca(filtered_df)


