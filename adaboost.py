import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from pca_embed import add_pca_features

def adaboost(pca_df, random_state=25):
    X_original = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=random_state
        )
    # Reset indexes to align correctly after splitting
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    model = AdaBoostClassifier(n_estimators=500, learning_rate=0.2, random_state=random_state)
    
    model.fit(X_train, y_train)     
    
    y_pred = model.predict(X_test)

    accuracy = (accuracy_score(y_test, y_pred))

    return accuracy

if __name__ == "__main__":
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, _ = add_pca_features(df, 37)


    model = adaboost(pca_df,  random_state=25)

    print(f'Accuracy: {model}')
    