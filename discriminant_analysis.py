import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from pca_embed import add_pca_features
from utils import filter_data_genre



def discriminant_analysis(df, test_size=0.2, random_state=42):
    X = df.drop(
        [
            "label",
            "title",
            "genre",
            "description",
            "description_t",
            "first_token",
            "last_token",
            "embedding",
        ],
        axis=1,
    )
    y = df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train Linear Discriminant Analysis model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict
    y_pred = lda.predict(X_test)

    # Print classification report
    print("Report on Embeddings data:")
    print(classification_report(y_test, y_pred))

def discriminant_analysis_pca(pca_df, test_size=0.2, random_state=42):
    X_original = pca_df.drop("genre", axis=1)
    y = pca_df["genre"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=test_size, random_state=random_state)

    # Initialize and train Linear Discriminant Analysis model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict using the original feature space
    y_pred = lda.predict(X_test)

    # Print classification report
    print("Report on PCA data:")
    print(classification_report(y_test, y_pred))

    # Apply PCA to visualize in 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    # Plot decision boundaries
    plt.figure(figsize=(10, 8))
    h = .02  # step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot data points
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test_encoded, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('Decision Boundaries in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/train_data_embed.csv")
   
    filtered_df = filter_data_genre(df, minimum_threshold=1000)
    pca_df = add_pca_features(filtered_df, n_components=37)
    
    #discriminant_analysis(filtered_df)
    discriminant_analysis_pca(pca_df)