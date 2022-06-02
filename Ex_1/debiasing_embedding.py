from typing import List, Dict, Tuple

import numpy as np

from skipGram import SkipGram

# Ideas from https://arxiv.org/pdf/1607.06520.pdf


def compute_bias_vector(sg_model: SkipGram) -> np.ndarray:
    """
    Compute a embedding representation of the notion of gender

    :param sg_model: trained SkipGram model
    :return: Vector which correspond to the embedding of the notion of gender
    """
    man_embedding = sg_model.get_embedding("man")
    woman_embedding = sg_model.get_embedding("woman")

    # Just check that we do have a model that has "man" and "woman" in its
    # vocabulary, otherwise, we should find other words that carry a strong
    # gender meaning
    assert man_embedding is not np.zeros(
        sg.emdedding_dim
    ) and woman_embedding is not np.zeros(sg.emdedding_dim)

    gender_embedding = woman_embedding - man_embedding
    return gender_embedding


def cosine_similarity(
    word_1: np.ndarray,
    word_2: np.ndarray,
) -> float:
    """
    Compute cosine similarities of 2 words given their embeddings

    :param word_1: array, embedding of first word
    :param word_2: array, embedding of second word
    :return: float, cosine similarity between the 2 words
    """
    norm_word_1 = np.linalg.norm(word_1)
    norm_word_2 = np.linalg.norm(word_2)
    return (word_1 @ word_2) / (norm_word_1 * norm_word_2 + 1e-7)


def get_gender_biases(
    sg_model: SkipGram, words_list: List[str], unbias: bool = False
) -> Dict[str, Tuple[float, str]]:
    """
    Determine to which gender a list of words may be biased

    :param sg_model: trained SkipGram model
    :param words_list: list of words (str) to be tested
    :param unbias: if True, will take the embedding of the word
    after having removed any gender bias using remove_word_bias_component function

    :return: dictionary, with words as key, and tuple as keys that contains:
        - similarity score with gender_embedding (result of compute_bias_vector function)
        - gender orientation ("woman_oriented" if similarity > 0, "man_oriented" otherwise)
    """
    gender_embedding = compute_bias_vector(sg_model)
    gender_biases = {}
    for word in words_list:
        if unbias:
            word_embedding = remove_word_bias_component(
                sg_model, word, gender_embedding
            )
        else:
            word_embedding = sg_model.get_embedding(word)
        gender_similarity = cosine_similarity(word_embedding, gender_embedding)
        gender_biases[word] = (
            gender_similarity,
            "woman_oriented" if gender_similarity > 0 else "man_oriented",
        )
    return gender_biases


def remove_word_bias_component(
    sg_model: SkipGram, word: str, gender_embedding: np.ndarray
) -> np.ndarray:
    """
    Removes gender bias from a word embedding

    :param sg_model: trained SkipGram model
    :param word: word from which we want to remove gender bias
    :param gender_embedding: result of compute_bias_vector function
    (vector which correspond to the embedding of the notion of gender)
    :return: new embedding of the word from which the gender bias has been removed.
    """
    word_embedding = sg_model.get_embedding(word)
    bias_projection = [
        (word_embedding @ gender_embedding) / (gender_embedding @ gender_embedding)
    ] * gender_embedding
    return word_embedding - bias_projection


if __name__ == "__main__":
    model_path = "mymodel.model"
    sg = SkipGram.load(model_path)
    surnames_list = ["armstrong", "will", "kirk", "mary"]
    misc_list = [
        "work",
        "president",
        "children",
        "science",
        "literature",
        "father",
        "mother",
        "god",
        "boy",
        "girl",
    ]

    surnames_biases = get_gender_biases(sg, surnames_list, unbias=False)
    misc_biases = get_gender_biases(sg, misc_list, unbias=False)

    surnames_biases_neutralize = get_gender_biases(sg, surnames_list, unbias=True)
    misc_biases_neutralize = get_gender_biases(sg, misc_list, unbias=True)
