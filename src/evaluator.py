import math
from typing import Iterable, Sequence, Set

import pandas as pd


ANNOTATOR_1_RANKINGS = [
    [2, 1, 4, 3, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [3, 1, 2, 4, 5], [1, 5, 4, 2, 3],
    [3, 2, 1, 4, 5], [3, 2, 1, 5, 4], [2, 4, 3, 1, 5], [1, 5, 2, 3, 4], [3, 2, 1, 4, 5],
    [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 3, 2, 4, 5], [1, 2, 3, 4, 5], [3, 1, 2, 4, 5],
    [3, 1, 2, 4, 5], [3, 1, 2, 4, 5], [1, 2, 5, 3, 4], [3, 2, 1, 4, 5], [3, 2, 1, 4, 5],
    [2, 3, 1, 4, 5], [1, 2, 3, 5, 4], [2, 1, 3, 5, 4], [1, 2, 3, 5, 4], [1, 2, 3, 4, 5],
    [2, 1, 3, 4, 5], [2, 3, 4, 5, 1], [2, 4, 3, 1, 5], [5, 1, 2, 4, 3], [2, 1, 4, 3, 5],
]

ANNOTATOR_2_RANKINGS = [
    [4, 3, 1, 5, 2], [2, 4, 3, 1, 5], [5, 4, 2, 3, 1], [1, 3, 2, 4, 5], [5, 1, 2, 4, 3],
    [1, 3, 2, 4, 5], [4, 2, 3, 1, 5], [2, 4, 3, 1, 5], [3, 4, 2, 1, 5], [4, 1, 2, 5, 3],
    [2, 4, 3, 5, 1], [4, 3, 2, 1, 5], [4, 2, 3, 1, 5], [3, 4, 2, 1, 5], [2, 4, 3, 1, 5],
    [3, 2, 4, 1, 5], [4, 2, 3, 1, 5], [4, 2, 5, 3, 1], [4, 2, 3, 1, 5], [1, 5, 2, 4, 3],
    [1, 3, 4, 5, 2], [4, 1, 3, 2, 5], [1, 3, 4, 2, 5], [1, 4, 3, 5, 2], [1, 4, 2, 5, 3],
    [1, 5, 2, 4, 3], [4, 3, 1, 2, 5], [1, 4, 2, 3, 5], [5, 1, 2, 4, 3], [1, 2, 3, 4, 5],
]


def _as_set(values: Iterable[int]) -> Set[int]:
    return set(int(value) for value in values)


def precision_at_k(predicted_ranking: Sequence[int], relevant_items: Iterable[int], k: int = 5) -> float:
    if k <= 0:
        return 0.0

    top_k = list(predicted_ranking)[:k]
    relevant = _as_set(relevant_items)

    if not top_k:
        return 0.0

    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(predicted_ranking: Sequence[int], relevant_items: Iterable[int], k: int = 5) -> float:
    relevant = _as_set(relevant_items)

    if not relevant:
        return 0.0

    top_k = set(list(predicted_ranking)[:k])
    hits = len(top_k.intersection(relevant))

    return hits / len(relevant)


def reciprocal_rank(predicted_ranking: Sequence[int], relevant_items: Iterable[int]) -> float:
    relevant = _as_set(relevant_items)

    for index, item in enumerate(predicted_ranking, start=1):
        if item in relevant:
            return 1 / index

    return 0.0


def average_precision_at_k(predicted_ranking: Sequence[int], relevant_items: Iterable[int], k: int = 5) -> float:
    relevant = _as_set(relevant_items)

    if not relevant:
        return 0.0

    score_sum = 0.0
    hits = 0

    for index, item in enumerate(list(predicted_ranking)[:k], start=1):
        if item in relevant:
            hits += 1
            score_sum += hits / index

    return score_sum / min(len(relevant), k)


def dcg_at_k(predicted_ranking: Sequence[int], relevant_items: Iterable[int], k: int = 5) -> float:
    relevant = _as_set(relevant_items)
    score = 0.0

    for index, item in enumerate(list(predicted_ranking)[:k], start=1):
        relevance = 1 if item in relevant else 0
        score += relevance / math.log2(index + 1)

    return score


def ndcg_at_k(predicted_ranking: Sequence[int], relevant_items: Iterable[int], k: int = 5) -> float:
    relevant = _as_set(relevant_items)

    if not relevant:
        return 0.0

    ideal_ranking = list(relevant)[:k]
    ideal_dcg = dcg_at_k(ideal_ranking, relevant, k)

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(predicted_ranking, relevant, k) / ideal_dcg


def expert_relevant_jobs(cv_id: int, top_n: int = 1, mode: str = "union") -> Set[int]:
    """
    Zwraca zbiór ofert uznanych przez ekspertów za relewantne dla danego CV.

    cv_id: numer CV od 1 do 30.
    top_n: ile najlepszych ofert każdego anotatora traktujemy jako relewantne.
    mode:
        - "union": wystarczy, że oferta jest w top_n u jednego anotatora,
        - "intersection": oferta musi być w top_n u obu anotatorów.
    """
    if cv_id < 1 or cv_id > 30:
        raise ValueError("cv_id musi być w zakresie 1-30 dla dostępnych anotacji eksperckich.")

    annotator_1_top = set(ANNOTATOR_1_RANKINGS[cv_id - 1][:top_n])
    annotator_2_top = set(ANNOTATOR_2_RANKINGS[cv_id - 1][:top_n])

    if mode == "union":
        return annotator_1_top.union(annotator_2_top)

    if mode == "intersection":
        return annotator_1_top.intersection(annotator_2_top)

    raise ValueError("mode musi mieć wartość 'union' albo 'intersection'.")


def evaluate_ranking(predicted_ranking: Sequence[int], relevant_items: Iterable[int], k_values=(1, 3, 5)) -> dict:
    metrics = {}

    for k in k_values:
        metrics[f"Precision@{k}"] = round(precision_at_k(predicted_ranking, relevant_items, k), 4)
        metrics[f"Recall@{k}"] = round(recall_at_k(predicted_ranking, relevant_items, k), 4)
        metrics[f"MAP@{k}"] = round(average_precision_at_k(predicted_ranking, relevant_items, k), 4)
        metrics[f"NDCG@{k}"] = round(ndcg_at_k(predicted_ranking, relevant_items, k), 4)

    metrics["MRR"] = round(reciprocal_rank(predicted_ranking, relevant_items), 4)

    return metrics


def summarize_metrics(results_df: pd.DataFrame, model_column: str = "Model") -> pd.DataFrame:
    metric_columns = [
        column for column in results_df.columns
        if column.startswith("Precision@")
        or column.startswith("Recall@")
        or column.startswith("MAP@")
        or column.startswith("NDCG@")
        or column == "MRR"
    ]

    summary = (
        results_df
        .groupby(model_column)[metric_columns]
        .mean()
        .round(4)
        .reset_index()
    )

    return summary
