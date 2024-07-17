from stopwordsiso import stopwords
import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from stopwordsiso import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic import BERTopic
from .tools import *

encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def run(df: pd.DataFrame,top_n_words=100, **kwargs):
    """
    Topic Modeling with BERTopic process
    """
    ### PROCESSED DF
    file_path = 'database/processed_hotel_reviews.csv'
    if os.path.isfile(file_path):
        processed_df = pd.read_csv(file_path, index_col=0)
    else:    
        processed_df = preprocess(df,type = 'sample')
    docs = processed_df[processed_df.index.isin(df.index)].processed_comment.values.astype('str')
    
    ### EMBEDDINGS
    encoder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    file_path = 'database/corpus_embeddings.npy'
    if os.path.isfile(file_path):
        corpus_embeddings = np.load(file_path)
        extracted_embedding = extract_embedding(df, corpus_embeddings)
    else:
        corpus_embeddings = embedding(docs,encoder, type='sample')
        df = df.reset_index()
        extracted_embedding = extract_embedding(df, corpus_embeddings)
    
    ### Set default values for parameters
    umap_defaults = {
        'n_neighbors': 15,
        'n_components': 5,
        'min_dist': 0.0,
        'metric': 'cosine',
        'random_state': 42
    }
    
    hdbscan_defaults = {
        'metric': 'euclidean',
        'min_cluster_size': 5,
        'min_samples': 1,
        'prediction_data': True
    }
    
    ### Override defaults with any provided kwargs
    umap_params = {**umap_defaults, **kwargs.get('umap_params', {})}
    hdbscan_params = {**hdbscan_defaults, **kwargs.get('hdbscan_params', {})}
    
    ### DIMENSIONALITY REDUCTION:
    umap_model = UMAP(**umap_params)
    
    ### CLUSTERING:
    hdbscan_model = HDBSCAN(**hdbscan_params)

    ### VECTORIZERS:
    stop_word_mul = stopwords(['vi','en'])
    vectorizer_model = CountVectorizer(stop_words=list(stop_word_mul), min_df=2, ngram_range=(1, 2))

    ### c-TF-IDF:
    ctfidf_model = ClassTfidfTransformer()
    
    ### REPRESENTATION:
    keybert_model = KeyBERTInspired()
    pos_model = PartOfSpeech('en_core_web_sm')
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    representation_model = {
        'KeyBERT': keybert_model,
        'MMR': mmr_model,
        'POS': pos_model
    }
    
    topic_model = BERTopic(
        embedding_model=encoder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        # hyperparameters
        top_n_words=top_n_words,
        verbose=True
    )
    topics, probs = topic_model.fit_transform(docs, extracted_embedding)
    return topic_model