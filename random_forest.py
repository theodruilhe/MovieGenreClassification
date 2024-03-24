import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from pca_embed import add_pca_features


def randomforest(pca_df, random_state=25, n_estimators=200, max_depth=20):

    X_original = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=random_state
    )
    # Reset indexes to align correctly after splitting
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        max_features="sqrt",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)

    return model, accuracy


if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df, 37)

    model, accuracy = randomforest(
        pca_df, n_estimators=100, max_depth=20, random_state=42
    )

    print(f"Accuracy: {accuracy*100:.2f}%")
