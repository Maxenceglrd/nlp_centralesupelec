from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import gensim.downloader as api

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


KEPT_POSTAG = ["ADV", "ADJ", "INTJ"]


class Featurizer:
    """
    Build the feature representation of a dataset
    """

    def __init__(self):
        self.aspect_category_encoder = LabelEncoder()
        self.polarity_label_encoder = LabelEncoder()

        self.sentences_vectorizer = TfidfVectorizer(stop_words="english")
        self.glove_model = self.load_glove_model()

    def get_features_and_labels(
        self, dataframe: pd.DataFrame, train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main function that returns, given a dataset, the features representation
        as well as the encoding of the labels

        :params:
        dataframe: pandas DataFrame that represents the dataset
        train: boolean, if True, means that we need to fit the TD-IDF and the LabelEncoder

        return:
        Tuple of arrays: (features, labels)
        """
        if train:
            labels = self.polarity_label_encoder.fit_transform(dataframe.polarity)
            sentences_vectors = self.sentences_vectorizer.fit_transform(
                dataframe.sentence
            )
            aspect_category_vectors = self.aspect_category_encoder.fit_transform(
                dataframe.aspect_category
            ).reshape(-1, 1)

        else:
            labels = self.polarity_label_encoder.transform(dataframe.polarity)
            sentences_vectors = self.sentences_vectorizer.transform(dataframe.sentence)
            aspect_category_vectors = self.aspect_category_encoder.transform(
                dataframe.aspect_category
            ).reshape(-1, 1)
        parser_vectors = self.get_parser_vectors(dataframe)
        features = hstack([sentences_vectors, aspect_category_vectors, parser_vectors])
        return features.toarray(), labels

    @staticmethod
    def load_glove_model():
        """
        Load a pretrained GloVe model
        """
        model = api.load("glove-wiki-gigaword-100")
        return model

    @staticmethod
    def transform_to_word_offsets(
        sentences: List[str], character_offsets: List[str]
    ) -> List[int]:
        """
        Function that transform character offset of a given word into a word offset

        :params:
        sentences: List of strings
        character_offsets: List of values of the form "min_offset:max_offset" corresponding
        to the character offsets of a the aspect term we are considering

        return
        words_offsets: List of integers representing the word offsets
        """
        words_offsets = []
        for sentence, character_offset in zip(sentences, character_offsets):
            min_offset, _ = list(map(int, character_offset.split(":")))
            words = sentence.split(" ")
            cumulated_offset = 0
            for i, word in enumerate(words):
                cumulated_offset += len(word)
                if min_offset <= cumulated_offset + i + 1:
                    words_offsets.append(i)
                    break
        return words_offsets

    def get_parser_vectors(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Get a polarisation score for each sentence

        :params:
        dataframe: pandas Dataframe representing the dataset

        return
        An array of dimension (number of sentences, 3) where each column
        correspond to a score relatively to a polarity (respectively positive,
        negative, neutral)
        """
        positive_embedding = self.glove_model["positive"]
        negative_embedding = self.glove_model["negative"]
        neutral_embedding = self.glove_model["neutral"]

        sentences = dataframe.sentence
        character_offsets = dataframe.character_offsets
        word_offsets = self.transform_to_word_offsets(sentences, character_offsets)
        features = []
        for sentence, word_offset in zip(sentences, word_offsets):
            sentence_parser_features = np.zeros(3)
            for i, token in enumerate(nlp(sentence)):
                distance_to_target_term = abs(i - word_offset)
                if token.pos_ in KEPT_POSTAG and distance_to_target_term < 5:
                    try:
                        word_embedding = self.glove_model[token.lemma_]
                        polarity = [
                            word_embedding @ positive_embedding,
                            word_embedding @ negative_embedding,
                            word_embedding @ neutral_embedding,
                        ]
                        best_polarity_index = np.argmax(polarity)
                        sentence_parser_features[best_polarity_index] += polarity[
                            best_polarity_index
                        ] / (distance_to_target_term + 1)
                    except KeyError:
                        pass
            features.append(sentence_parser_features)
        return np.array(features)
