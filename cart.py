import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pca_embed import add_pca_features


def cart(pca_df, ploting=False):

    X = pca_df.iloc[:, :-1]  # Features
    y = pca_df['genre']  # Target variable


    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating Decision Tree classifier object
    classifier = DecisionTreeClassifier()

    # Training Decision Tree Classifier
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Calculating the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    if ploting==True:
        # Visualizing the Decision Tree
        plt.figure(figsize=(20,10))
        plot_tree(classifier, filled=True, feature_names=X.columns, class_names=classifier.classes_)
        plt.show()
    return(classifier)

if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")
    pca_df, _ = add_pca_features(df, 37)
    model=cart(pca_df,False)
