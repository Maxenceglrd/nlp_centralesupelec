from __future__ import division
import argparse
import pickle
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Union, Iterator

import numpy as np
import pandas as pd
from scipy.special import expit
import spacy

__authors__ = ["Marien Chenaud", "Maxence Gélard"]
__emails__ = [
    "marien.chenaud@student-cs.fr",
    "maxence.gelard@student-cs.fr",
]

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


def text2sentences(path: str) -> List[List[str]]:
    """
    Process a given text by splitting each sentence into multiple tokens.
    Performs some text-preprocessing (lemmatization, stop words removal)

    :param path: path from which we take the dataset
    :return: list of processed sentences, each sentences being a list of tokens (strings)
    """
    sentences = []
    with open(path) as f:
        for line in f:
            tokens = nlp(line.lower())
            sentences.append(
                [
                    token.lemma_
                    for token in tokens
                    if token.text not in nlp.Defaults.stop_words
                    and not (token.is_punct or token.is_digit)
                    and token.text != "\n"
                ]
            )
    return sentences


def load_pairs(path: str) -> Iterator[Tuple[str, str, float]]:
    """
    Load an annotated dataset of words similarities

    :param path: path of the dataset
    :return: List of tuples of the form (word 1, word 2, similarity(word 1, word 2))
    """
    data = pd.read_csv(path, delimiter="\t")
    word_pairs = zip(data["word1"], data["word2"], data["similarity"])
    return word_pairs


def build_dataset(
    sentences: List[List[str]], min_count: int
) -> Tuple[Set[str], Dict[str, int], Dict[str, int]]:
    """
    Build the training dataset

    :param sentences: output of text2sentences function
    list of processed sentences, each sentences being a list of tokens (strings)
    :param min_count: minimum occurence of a word for it to be included
    in the vocabulary
    :return: Tuple composed of 3 elements:
        - Vocabulary: set of strings-
        - Word to id: dictionary that maps words (str) to ids (int)
        - Id to word: dictionary that maps ids (int) to words (str)
    """
    vocabulary = set()
    word_to_id = {}
    word_count = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            word_count[word] += 1
            if word not in vocabulary and word_count[word] >= min_count:
                word_to_id[word] = len(vocabulary)
                vocabulary.add(word)
    return (
        vocabulary,
        word_to_id,
        {w: count for w, count in word_count.items() if count >= min_count},
    )


class SkipGram:
    def __init__(
        self,
        sentences: List[List[str]],
        emdedding_dim: int = 100,
        negative_sample_rate: int = 15,
        window_size: int = 10,
        min_count: int = 2,
        n_epochs: int = 10,
        lr: float = 0.001,
    ):
        self.train_set = sentences  # set of sentences

        # self.vocabulary: list of valid words
        self.vocabulary, self.word_to_id, self.word_count = build_dataset(
            sentences, min_count
        )

        # self.word_to_id: word to ID mapping
        self.vocabulary_size = len(self.vocabulary)

        # self.id_to_word: ID to word mapping
        self.id_to_word = {word_id: word for word, word_id in self.word_to_id.items()}

        self.negative_sample_rate = negative_sample_rate
        self.negative_sample_probabilities = self.build_negative_sample_probabilities()

        self.emdedding_dim = emdedding_dim
        self.word_embedding, self.context_embedding = self.initialize_embedding()

        self.window_size = window_size

        self.train_words = 0
        self.accumulated_loss = 0
        self.loss = []

        self.lr = lr
        self.n_epochs = n_epochs

    def initialize_embedding(self, init_type="uniform") -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize words and contexts embedding matrices

        :param init_type: type of initialization, either "normal" or "uniform"
        :return: two arrays, respectively words and contexts embeddings matrices
        whose shape is (vocabulary size, embedding dimension)
        """
        if init_type == "normal":
            word_embedding = np.random.randn(self.vocabulary_size, self.emdedding_dim)
            context_embedding = np.random.randn(
                self.vocabulary_size, self.emdedding_dim
            )
        else:
            word_embedding = np.random.random((self.vocabulary_size, self.emdedding_dim))
            context_embedding = np.random.random((self.vocabulary_size, self.emdedding_dim))
        return word_embedding, context_embedding

    def build_negative_sample_probabilities(self) -> np.ndarray:
        """
        Initialize negative sampling probabilities arrays

        :return: Array of probabilities corresponding to the negative sample probability
        for each word in the vocabulary
        """
        exp_count = np.array([count ** (3 / 4) for count in self.word_count.values()])
        normalizing_factor = np.sum(exp_count)
        return np.array(
            [exp_count[i] / normalizing_factor for i in range(len(exp_count))]
        )

    def sample(self, omit: Set[int]) -> List[int]:
        """
        :param omit: set of word ids (word id, context id)
        :return:
        List of sampled negative words, omitting those in set omit
        """
        negative_ids = []
        words = list(self.id_to_word.keys())
        while len(negative_ids) < self.negative_sample_rate:
            negative_id_candidate = int(
                np.random.choice(
                    words,
                    size=1,
                    p=self.negative_sample_probabilities,
                )
            )
            if negative_id_candidate not in omit:
                negative_ids.append(negative_id_candidate)
        return negative_ids

    def train(self) -> None:
        """
        Main train loop of SkipGram models
        """
        loss = []
        for i in range(self.n_epochs):
            print(f"Epoch {i}")
            for counter, sentence in enumerate(self.train_set):
                sentence = list(filter(lambda w: w in self.vocabulary, sentence))

                for word_pos, word in enumerate(sentence):
                    word_idx = self.word_to_id[word]
                    window_size = np.random.randint(self.window_size) + 1
                    start = max(0, word_pos - window_size)
                    end = min(word_pos + window_size + 1, len(sentence))

                    for context_word in sentence[start:end]:
                        ctxt_id = self.word_to_id[context_word]
                        if ctxt_id == word_idx:
                            continue
                        negative_ids = self.sample({word_idx, ctxt_id})
                        self.train_word(word_idx, ctxt_id, negative_ids)
                        self.train_words += 1

            current_loss = self.accumulated_loss / self.train_words
            loss.append(current_loss)
            print(f"Epoch {i}: loss: {current_loss}")
            self.train_words = 0
            self.accumulated_loss = 0.0

        self.loss = loss

    @staticmethod
    def compute_loss(
        word_embedding: np.ndarray,
        context_embedding: np.ndarray,
        negative_embedding: np.ndarray,
    ) -> float:
        """
        Compute the partial loss corresponding to the couple (word, context)

        :param word_embedding: embedding (from word embedding matrix) of the considered word
        :param context_embedding: embedding (from context embedding matrix) of the considered context
        :param negative_embedding: embeddings (from context embedding matrix) of the considered negative samples
        :return: Contribution of the couple (word, context) to the current loss
        """
        negative_samples_loss = np.sum(
            np.log(expit(-negative_embedding @ word_embedding))
        )
        context_loss = np.log(expit(context_embedding @ word_embedding))
        return -(context_loss + negative_samples_loss)

    def train_word(self, word_id, context_id, negative_ids) -> None:
        """
        Train step for a single (word, context) pair. Computes gradients
        and does a single gradient descent step.

        :param word_id: id of the considered word
        :param context_id: id of the considered context word for the chosen word
        :param negative_ids: ids of the negative samples for the considered for
        :return: None
        """
        word_embedding = self.word_embedding[word_id]
        negative_embedding = np.array([self.word_embedding[i] for i in negative_ids])
        context_embedding = self.word_embedding[context_id]
        loss = self.compute_loss(word_embedding, context_embedding, negative_embedding)

        grad_loss_x_context = -context_embedding * expit(
            -context_embedding @ word_embedding
        )
        grad_loss_x_negative_samples = negative_embedding * expit(
            negative_embedding @ word_embedding
        ).reshape(-1, 1)
        grad_loss_x = grad_loss_x_context + np.sum(grad_loss_x_negative_samples)

        grad_loss_y = -word_embedding * expit(-context_embedding @ word_embedding)

        self.word_embedding[word_id, :] -= self.lr * grad_loss_x
        self.context_embedding[context_id, :] -= self.lr * grad_loss_y

        for i in range(len(negative_ids)):
            grad_loss_z_context = word_embedding * expit(
                negative_embedding[i] @ word_embedding
            )
            self.context_embedding[negative_ids[i], :] -= self.lr * grad_loss_z_context

        self.accumulated_loss += loss

    def get_embedding(self, word: str, combine_context: bool = False) -> np.ndarray:
        """
        Get the embedding of a word from the word embedding matrix

        :param word: str
        :param combine_context: if True, also use the context embedding
        and return the mean of word and context embedding instead.
        :return: Vector corresponding to the embedding of the word
        """
        word_id = self.word_to_id.get(word, -1)
        if word_id == -1:
            return np.zeros(self.emdedding_dim)
        else:
            if combine_context:
                return (
                    self.word_embedding[word_id, :] + self.context_embedding[word_id, :]
                ) / 2
            else:
                return self.word_embedding[word_id, :]

    def similarity(
        self,
        word_1: Union[str, np.ndarray],
        word_2: Union[str, np.ndarray],
        embedding: bool = False,
        combine_context: bool = False,
    ):
        """
        computes similiarity between the two words. unknown words are mapped to one common vector

        :param word_1: str or its embedding
        :param word_2:str or its embedding
        :param embedding: bool if True word_1 and word_2 are not string but already embeddings
        :param combine_context: if True, also use the context embedding
        and return the mean of word and context embedding instead.
        :return: a float in [0,1] indicating the similarity (the higher the more similar)
        """
        if not embedding:
            word_1 = self.get_embedding(nlp(word_1)[0].lemma_, combine_context)
            word_2 = self.get_embedding(nlp(word_2)[0].lemma_, combine_context)
        norm_word_1 = np.linalg.norm(word_1)
        norm_word_2 = np.linalg.norm(word_2)
        return 0.5 * (1 + (word_1 @ word_2) / (norm_word_1 * norm_word_2 + 1e-7))

    @staticmethod
    def load(path: str):
        """
        Load a previously trained SkipGram model

        :param path: path of the saved model
        :return: Saved SkipGram model
        """
        word_to_id_path = path + "_word_to_id.pkl"
        embedding_path = path + "_embedding.pkl"
        context_embedding_path = path + "_context_embedding.pkl"

        with open(word_to_id_path, "rb") as f:
            word_to_id = pickle.load(f)
        with open(embedding_path, "rb") as f:
            word_embedding = pickle.load(f)
        with open(context_embedding_path, "rb") as f:
            context_embedding = pickle.load(f)

        sg_model = SkipGram(sentences=[])
        sg_model.word_to_id = word_to_id
        sg_model.word_embedding = word_embedding
        sg_model.context_embedding = context_embedding
        sg_model.emdedding_dim = word_embedding.shape[1]
        return sg_model

    def save(self, path: str) -> None:
        """
        Saved the current instance of SkipGram model

        :param path: path used to save the model
        """
        word_to_id_path = path + "_word_to_id.pkl"
        embedding_path = path + "_embedding.pkl"
        context_embedding_path = path + "_context_embedding.pkl"

        with open(word_to_id_path, "wb") as f:
            pickle.dump(self.word_to_id, f)
        with open(embedding_path, "wb") as f:
            pickle.dump(self.word_embedding, f)
        with open(context_embedding_path, "wb") as f:
            pickle.dump(self.context_embedding, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        help="path containing training data",
        required=False,
        default="train.txt",
    )
    parser.add_argument(
        "--model",
        help="path to store/read model (when training/testing)",
        # required=True,
        required=False,
        default="mymodel.model",
    )
    parser.add_argument("--test", help="enters test mode", action="store_true")

    opts = parser.parse_args()
    if not opts.test:
        train_sentences = text2sentences(opts.text)
        sg = SkipGram(train_sentences)
        sg.train()
        sg.save(opts.model)

    else:
        pairs = load_pairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(sg.similarity(a, b))
