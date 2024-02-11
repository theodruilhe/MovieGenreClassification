# Movie Genre Classification

![AI generatad banner](./banner.png)

This project harnesses machine learning and natural language processing to
categorize movies into genres based on their IMDb descriptions. Employing a
robust dataset, the study undertakes tokenization and embedding for preprocessing,
followed by PCA for dimensionality reduction. With K-Means clustering, we uncover
distinct groupings within the movies, including a unique cluster characterized by
singular movie descriptions. The project leverages a Classification Tree and Linear
Discriminant Analysis (LDA), achieving an accuracy of 60% with LDA in genre prediction.

## Installation

After cloning this repo, run `pip install -r requirements.txt`
Then, in order for spacy to tokenize, run `python3 -m spacy download en_core_web_sm`

## Data

Download the Data on [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb/download?datasetVersionNumber=1), from the [Genre Classification Dataset IMDb page](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb), and move the files into a `data/` folder.

## Data Preprocessing

To create the tokens and embeddings from the dataset, use the `main()` function from the file `data_pipeline`, for example:

```python
from data_pipeline import main
from data_pipeline import merge_train_test
merge_train_test(
        "data/train_data.txt", "data/test_data_solution.txt", "data/full_data.txt"
    )
df = main("data/full_data.txt", save_file=True)
```

Depending on your computing power, the function may take some time to execute, for
example:

- ~ 13 minutes on an Apple M1 chip
- ~ 25 minutes on a 1,4 GHz Intel Core i5 four cores

Then the file `data/full_data_embed.csv` is created.

## Usage

### Inference App

You can try our models by running the following command on your terminal:

```bash
streamlit run inference.py
```

## Explore the different methods

### PCA

Contains various functions to

1. Compute principal components from the embeddings (`add_pca_features()`)
2. Plot the data on the first two components (`plot_pca()`)
3. Plot the scree plot (`scree_plot()`)

#### Example usage

Run this code **after** running the `main()` function from `data_pipeline.py`:

```python
# Import necessary packages and functions
import pandas as pd
from pca_embed import add_pca_features, scree_plot, plot_pca

# Read the data
df = pd.read_csv("data/full_data_embed.csv")

# Compute Principal Components and get FataFrame and pca model
pca_df, pca_model = add_pca_features(df, n_components=37)

# Plot the Scree Plot
scree_plot(pca_model)

# Plot the data colored by genre
plot_pca(pca_df)
```

### K-Means clustering

Contains various functions to

1. Plot the Elbow curve to choose the number of clusters (`elbow_method()`)
2. Compute clusters and return a DataFrame with data and clusters (`create_clustered_df()`)
3. Plot the data on the first two components, colored by cluster (`visualize_clusters()`)
4. Output a simple analysis of the clusters: genre distribution per cluster, as
a table and pie charts (`cluster_analysis()`)
5. Get random movie samples from a given cluster (`show_sample_from_cluster()`)

#### Example usage

```python
# Import necessary packages and functions
import pandas as pd
from utils import create_clustered_col
from pca_embed import add_pca_features
from clustering import create_clustered_df, visualize_clusters, cluster_analysis, elbow_method

# Load the data
df = pd.read_csv("data/full_data_embed.csv")

# Compute Principal Components
pca_df, _ = add_pca_features(df)

# Compute Clusters
full_df = create_clustered_df(pca_df, n_clusters=8)

# Plot Elbow Curve
elbow_method(pca_df.drop(columns=["genre"]), max_clusters=15)

# Plot data colored by cluster on the first two components
visualize_clusters(full_df)

# Output Cluster analysis
cluster_analysis(full_df)

# Output 10 random movies from cluster 0
cluster_column = create_clustered_col(full_df, df)
n_sample = 10
cluster = 0
desc = show_sample_from_cluster(cluster_column, cluster, n_sample)

# desc to latex table
print(
    desc[["title", "genre", "description"]].to_latex(
        index=False, caption="Random elements from cluster 0"
    )
)
```

### Classification Tree

The function `cart()` create a classification tree, train it on a training
sample (80% of the data), and test it on a test sample. Returns the classifier,
print out accuracy and plot the Tree.

#### Example usage

```python
# Import necessary packages and functions
import pandas as pd
from pca_embed import add_pca_features
from cart import cart

# Load the data
df = pd.read_csv("data/full_data_embed.csv")

# Add pca components
pca_df, _ = add_pca_features(df, 37)

# Train the model and plot the Tree
model = cart(full_df, True, random_state=29)
```

### Linear Discriminant Analysis

We implemented various functions to

1. Run linear discriminant analysis on the data (with 4 components) and output
the model and the transformed data (`discriminant_analysis_pca()`)
2. Plot the transformed data on the first two discriminant variables, colored by
genre (`plot_lda()`)
3. Output the cumulative explained variance (`show_explained_variance()`)

#### Example Usage

```python
# Import necessary packages and functions
import pandas as pd
from pca_embed import add_pca_features
from discriminant_analysis import discriminant_analysis_pca, show_explained_variance, plot_lda

# Load the data
df = pd.read_csv("data/full_data_embed.csv")

# Compute Principal Components
pca_df, _ = add_pca_features(df, n_components=37)

# Perform lda
lda, train_da, test_da = discriminant_analysis_pca(pca_df, heatmap=False)

# Show explained variance
show_explained_variance(lda)

# Plot transformed data
plot_lda(pca_df, lda)
```
