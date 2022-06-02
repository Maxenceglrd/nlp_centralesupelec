from typing import Tuple, Dict, List

import numpy as np

from skipGram import load_pairs, SkipGram


def get_similar_words(sg: SkipGram, ref_word: str) -> Tuple[Dict[float, str], str]:
    """
    Get all the similarity scores with a given word and retrieve best matching word

    :param sg: trained SkipGram model
    :param ref_word: str, word from which we want to extract similar word
    :return: Tuple containing:
        - Dictionary, mapping similarity score to word
        - Best matching word for ref_word
    """
    similar_words_dict = {}
    for word in sg.word_to_id.keys():
        if word != ref_word:
            similar_words_dict[sg.similarity(word, ref_word)] = word
    best_value = 0
    best_word = None
    for sim, w in similar_words_dict.items():
        if sim > best_value:
            best_value = sim
            best_word = w
    return similar_words_dict, best_word


def get_sorted_similarities(
    score_to_word: Dict[float, str], reverse: bool = False
) -> List[Tuple[float, str]]:
    """
    Helper function that allows to sort the dictionary returned
    by get_similar_words by scores

    :param score_to_word: Dictionary, mapping similarity score to word,
    returned by get_similar_words
    :param reverse: If true, scores are sorted in decreasing order
    :return: List of tuple (similarity, word) sorted by score
    """
    return sorted(
        [(s, w) for s, w in score_to_word.items()], key=lambda x: x[0], reverse=reverse
    )


def compute_similarity_mean_error(
    model_path: str = "mymodel.model", test_path: str = "simlex.csv"
) -> float:
    """
    Compute a score error between true similarties and predicted similarity

    :param model_path: path to a saved model
    :param test_path: path to the test dataset
    :return: float, error score on the test dataset
    """
    pairs = load_pairs(test_path)

    sg = SkipGram.load(model_path)
    ref_similarities = []
    pred_similarities = []
    for a, b, ref_similarity in pairs:
        ref_similarities.append(ref_similarity)
        pred_similarities.append(sg.similarity(a, b, combine_context=False))

    max_ref_sim = np.max(ref_similarities)
    min_ref_sim = np.min(ref_similarities)

    for i, s in enumerate(ref_similarities):
        ref_similarities[i] = (s - min_ref_sim) / (max_ref_sim - min_ref_sim)

    error = np.mean(
        [(s - s_ref) ** 2 for (s, s_ref) in zip(pred_similarities, ref_similarities)]
    )
    return error


def compute_pearson_correlation(
    model_path: str = "mymodel.model", test_path: str = "simlex.csv"
) -> float:
    """
    Compute the pearson correlation between true similarties and predicted similarity

    :param model_path: path to a saved model
    :param test_path: path to the test dataset
    :return: float, pearson correlation
    """
    pairs = load_pairs(test_path)

    sg = SkipGram.load(model_path)
    ref_similarities = []
    pred_similarities = []
    for a, b, ref_similarity in pairs:
        ref_similarities.append(ref_similarity)
        pred_similarities.append(sg.similarity(a, b, combine_context=False))

    ref_similarities = np.array(ref_similarities)
    pred_similarities = np.array(pred_similarities)

    ref_similarities_centered = ref_similarities - np.mean(ref_similarities)
    pred_similarities_centered = pred_similarities - np.mean(pred_similarities)

    pearson_correlation = np.sum(
        ref_similarities_centered * pred_similarities_centered
    ) / (
        np.sqrt(
            np.sum(ref_similarities_centered ** 2)
            * np.sum(pred_similarities_centered ** 2)
        )
    )
    return pearson_correlation


if __name__ == "__main__":
    pearson_correlation = compute_pearson_correlation()
