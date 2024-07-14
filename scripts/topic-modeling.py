import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from stopwordsiso import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech


encoder=SentenceTransformer('distiluse-base-multilingual-cased-v1')
encoder.max_seq_length= 512
stop_word_mul = stopwords(['vi','en'])
umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model=HDBSCAN(metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model=CountVectorizer(stop_words=list(stop_word_mul), min_df=2, ngram_range=(1,2))
keybert_model=KeyBERTInspired()
pos_model=PartOfSpeech('en_core_web_sm')
mmr_model=MaximalMarginalRelevance(diversity=0.3)
representation_model={
    'KeyBERT':keybert_model,
    'MMR':mmr_model,
    'POS':pos_model
}

def run(sample):
    """
    Topic Modeling with BERTopic process
    """
    ### EMBEDDINGS
    try:
        corpus_embeddings = pd.read_csv(f'database/corpus_embeddings_{language}')        
    except: 
        corpus_embeddings = encoder.encode(sample, show_progress_bar=True)
    ### DIMENSIONALITY REDUCTION:
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    ### CLUSTERING:
    hdbscan_model=HDBSCAN(metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    ### VECTORIZERS:
    stop_word_mul = stopwords(['vi','en'])
    vectorizer_model=CountVectorizer(stop_words=list(stop_word_mul), min_df=2, ngram_range=(1,2))

    ### c-TF-IDF:
    