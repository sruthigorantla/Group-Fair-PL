# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np

import utils.plackettluce as pl
import utils.ranking as rnk


def ideal_metrics(data_split, rank_weights, labels):
    cutoff = rank_weights.size
    result = np.zeros(data_split.num_queries())
    for qid in range(data_split.num_queries()):
        q_labels = data_split.query_values_from_vector(qid, labels)
        ranking = rnk.cutoff_ranking(-q_labels, cutoff)
        result[qid] = np.sum(rank_weights[: ranking.size] * q_labels[ranking])
    return result


def evaluate_expected(
    data_split,
    scores,
    rank_weights,
    labels,
    ideal_metrics,
    num_samples,
    group_membership=None,
    fairness_constraints=None,
    prefix_fairness_constraints=None,
    postprocess_algorithms=None,
    is_last_epoch=False,
):
    results_dictionary = {}
    for alg in postprocess_algorithms:
        cutoff = rank_weights.size
        n_groups = len(np.unique(group_membership))
        result = 0.0
        query_normalized_result = 0.0
        query_proportions = np.zeros((data_split.num_queries(), n_groups, cutoff))
        for qid in range(data_split.num_queries()):
            q_scores = data_split.query_values_from_vector(qid, scores)
            q_labels = data_split.query_values_from_vector(qid, labels)
            q_group_membership = data_split.query_values_from_vector(
                qid, group_membership
            )
            group_ids = []
            for j in range(n_groups):
                group_ids.append(np.where(q_group_membership == j)[0])

            # postprocess here
            if alg == "none":
                sampled_rankings = pl.gumbel_sample_rankings(
                    log_scores=q_scores,
                    n_samples=num_samples,
                    cutoff=cutoff,
                    group_ids=np.arange(len(q_scores)),
                )[0]
            else:
                sampled_rankings = pl.sample_group_fair_rankings(
                    log_scores=q_scores,
                    n_samples=num_samples,
                    group_ids=group_ids,
                    fairness_constraints=fairness_constraints,
                    prefix_fairness_constraints=prefix_fairness_constraints,
                    cutoff=cutoff,
                    postprocess=alg,
                )
            ### this block is to compute fraction of items from group in the cutoff

            if is_last_epoch:
                group_ids = [
                    np.where(q_group_membership == j)[0] for j in range(n_groups)
                ]

                for j in range(n_groups):
                    query_proportions[qid][j] = (
                        np.sum(
                            np.in1d(sampled_rankings, group_ids[j]).reshape(
                                (num_samples, cutoff)
                            ),
                            axis=0,
                        )
                    ) / num_samples

            ### this is to compute ndcg
            q_result = np.mean(
                np.sum(
                    rank_weights[None, : sampled_rankings.shape[1]]
                    * q_labels[sampled_rankings],
                    axis=1,
                ),
                axis=0,
            )
            result += q_result
            if ideal_metrics[qid] != 0:
                query_normalized_result += q_result / ideal_metrics[qid]
        result /= data_split.num_queries()
        query_normalized_result /= data_split.num_queries()
        normalized_result = result / np.mean(ideal_metrics)
        if is_last_epoch:
            results_dictionary[alg] = [
                result,
                normalized_result,
                query_normalized_result,
                query_proportions.tolist(),
            ]
        else:
            results_dictionary[alg] = [
                result,
                normalized_result,
                query_normalized_result,
            ]

    return results_dictionary


def compute_results(
    data_split,
    model,
    rank_weights,
    labels,
    ideal_metrics,
    num_samples,
    group_membership,
    fairness_constraints,
    prefix_fairness_constraints,
    postprocess_algorithms,
    is_last_epoch,
):
    scores = model(data_split.feature_matrix).numpy()[:, 0]

    return compute_results_from_scores(
        data_split,
        scores,
        rank_weights,
        labels,
        ideal_metrics,
        num_samples,
        group_membership,
        fairness_constraints,
        prefix_fairness_constraints,
        postprocess_algorithms,
        is_last_epoch,
    )


def compute_results_from_scores(
    data_split,
    scores,
    rank_weights,
    labels,
    ideal_metrics,
    num_samples,
    group_membership,
    fairness_constraints,
    prefix_fairness_constraints,
    postprocess_algorithms,
    is_last_epoch,
):
    results_dictionary = evaluate_expected(
        data_split,
        scores,
        rank_weights,
        labels,
        ideal_metrics,
        num_samples,
        group_membership,
        fairness_constraints,
        prefix_fairness_constraints,
        postprocess_algorithms,
        is_last_epoch,
    )

    aggregated_results = []
    if is_last_epoch:
        for alg in postprocess_algorithms:
            E, N_E, QN_E, P = (
                results_dictionary[alg][0],
                results_dictionary[alg][1],
                results_dictionary[alg][2],
                results_dictionary[alg][3],
            )

            aggregated_results.append(
                {
                    "postprocess algorithm": alg,
                    "expectation": E,
                    "normalized expectation": N_E,
                    "query normalized expectation": QN_E,
                    "query group proportions": P,
                }
            )
    else:
        for alg in postprocess_algorithms:
            E, N_E, QN_E = (
                results_dictionary[alg][0],
                results_dictionary[alg][1],
                results_dictionary[alg][2],
            )

            aggregated_results.append(
                {
                    "postprocess algorithm": alg,
                    "expectation": E,
                    "normalized expectation": N_E,
                    "query normalized expectation": QN_E,
                }
            )

    return aggregated_results


def evaluate_fairness(data_split, model, rank_weights, labels, num_samples):
    cutoff = rank_weights.size
    scores = model(data_split.feature_matrix).numpy()[:, 0]

    result = 0.0
    squared_result = 0.0
    for qid in range(data_split.num_queries()):
        q_scores = data_split.query_values_from_vector(qid, scores)
        q_labels = data_split.query_values_from_vector(qid, labels)
        if np.sum(q_labels) > 0 and q_labels.size > 1:
            sampled_rankings = pl.gumbel_sample_rankings(
                q_scores, num_samples, cutoff=cutoff
            )[0]

            q_n_docs = q_labels.shape[0]
            q_cutoff = min(cutoff, q_n_docs)
            doc_exposure = np.zeros(q_n_docs, dtype=np.float64)
            np.add.at(doc_exposure, sampled_rankings, rank_weights[:q_cutoff])
            doc_exposure /= num_samples

            swap_reward = doc_exposure[:, None] * q_labels[None, :]

            q_result = np.mean((swap_reward - swap_reward.T) ** 2.0)
            q_result *= q_n_docs / (q_n_docs - 1.0)

            q_squared = np.mean(np.abs(swap_reward - swap_reward.T))
            q_squared *= q_n_docs / (q_n_docs - 1.0)

            result += q_result
            squared_result += q_squared
    result /= data_split.num_queries()
    squared_result /= data_split.num_queries()
    return result, squared_result


def compute_fairness_results(data_split, model, rank_weights, labels, num_samples):
    absolute, squared = evaluate_fairness(
        data_split, model, rank_weights, labels, num_samples
    )
    return {
        "expectation absolute": absolute,
        "expectation squared": squared,
    }
