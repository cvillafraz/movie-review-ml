from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors


def _return_word_in_vocab(
    word: str, model: Word2Vec | KeyedVectors, num_features: int = 100
) -> np.ndarray:
    """
    This function checks if a word is in the vocabulary of a pre-trained Word2Vec
    model.

    Args:
        word : str
            A word that needs to be checked.

        model : Word2Vec | KeyedVectors
            A pre-trained Word2Vec model that will be used to check if the word
            is in the vocabulary.

    Returns:
        The word embedding if word is present in the model vocab. Else an numpy
        array of zeros.
    """
    try:
        if isinstance(model, Word2Vec):
            return model.wv[word]
        else:
            return model[word]
    except KeyError:
        return np.zeros(num_features)


def vectorizer(
    corpus: List[List[str]], model: Word2Vec | KeyedVectors, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec | KeyedVectors
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """
    corpus_vectors = np.zeros((len(corpus), num_features))
    for i, sentence in enumerate(corpus):
        embedding_list = [
            _return_word_in_vocab(word, model, num_features) for word in sentence
        ]
        sentence_embedding = np.mean(embedding_list, axis=0)
        corpus_vectors[i] = sentence_embedding
    return corpus_vectors
