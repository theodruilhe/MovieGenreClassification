# MovieGenreClassification

Academic project on Movie Genre Classification

## Installation

After cloning this repo, run `pip install -r requirements.txt`
Then, in order for spacy to tokenize, run `python3 -m spacy download en_core_web_sm`

## Data

Download the Data on [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb/download?datasetVersionNumber=1), from the [Genre Classification Dataset IMDb page](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb), and move the files into a `data/` folder.

## Usage

To create the tokens and embeddings from the dataset, use the `main()` function from the file `data_pipeline`, for example:

```python
from data_pipeline import main
train_data = main(data/train_data.txt, save=True)
```

Then the file `data/train_data_embed.csv` is created.
