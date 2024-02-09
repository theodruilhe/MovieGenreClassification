import numpy as np
import pandas as pd
import spacy
import streamlit as st
from gensim.models import Word2Vec
from tqdm import tqdm

from cart import cart
from data_pipeline import tokenize_col
from discriminant_analysis import discriminant_analysis_pca
from pca_embed import add_pca_features
from utils import filter_data_genre

# nlp = spacy.load("en_core_web_sm")


def inference_pipeline_da(text, nlp, pca_param, pred_model, w2v_model):
    tqdm.pandas()

    data = [text]
    df = pd.DataFrame(data, columns=["description"])

    data = tokenize_col(df, ["description"], nlp, remove_stop=True)

    embeddings = w2v_model.wv[data.description_t[0]]
    mean_embedding = np.nanmean(embeddings, axis=0)

    # inference with PCA

    projected_embedding = pca_param.transform(mean_embedding.reshape(1, -1))

    # inference with LDA
    prediction = pred_model.predict(projected_embedding)

    return mean_embedding, projected_embedding, prediction[0]


def train_all():
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv("data/full_data_embed.csv")

    pca_df, pca_param = add_pca_features(df, n_components=37)

    lda_model, _, _ = discriminant_analysis_pca(pca_df)

    cart_model = cart(pca_df, plotting=False, random_state=29)

    return nlp, pca_param, lda_model, cart_model


@st.cache_resource
def load_models():
    nlp, pca_param, lda_model, cart_model = train_all()
    w2v_model = Word2Vec.load("data/description_embedding.model")
    return nlp, pca_param, lda_model, cart_model, w2v_model


if __name__ == "__main__":
    nlp, pca_param, lda_model, cart_model, w2v_model = load_models()

    st.title("Genre Classifier")
    st.write(
        "This app uses a Linear Discriminant Analysis model to classify the genre of a movie based on its description."
    )

    text = st.text_area("Enter a movie description")

    output = st.empty()

    if st.button("Classify with LDA"):
        if text:
            mean_embedding, projected_embedding, prediction = inference_pipeline_da(
                text, nlp, pca_param, lda_model, w2v_model
            )
            output.write(f"Predicted genre: {prediction}")
        else:
            output.write("Please enter a movie description")
    if st.button("Classify with CART"):
        if text:
            mean_embedding, projected_embedding, prediction = inference_pipeline_da(
                text, nlp, pca_param, cart_model, w2v_model
            )
            output.write(f"Predicted genre: {prediction}")
        else:
            output.write("Please enter a movie description")
