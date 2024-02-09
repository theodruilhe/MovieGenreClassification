import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

from clustering import add_pca_features, create_clustered_df
from pca_embed import add_pca_features

# things needed to add clusters as info
from utils import create_clustered_col, get_random_elements_from_cluster


def cart(pca_df, plotting=False, random_state=29):

    X = pca_df.loc[:, pca_df.columns != "genre"]  # Features
    y = pca_df["genre"]  # Target variable

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Creating Decision Tree classifier object
    classifier = DecisionTreeClassifier(max_depth=8)

    # Training Decision Tree Classifier
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Calculating the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    if plotting == True:
        # Visualizing the Decision Tree
        plt.figure(figsize=(20, 10))
        plot_tree(
            classifier,
            filled=True,
            feature_names=X.columns,
            class_names=classifier.classes_,
        )
        plt.show()
    return classifier


if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df, 37)

    # adding the model appartenance
    full_df = create_clustered_df(pca_df, n_clusters=8)
    cluster_column = create_clustered_col(full_df, pca_df)

    model = cart(full_df, False, random_state=29)
