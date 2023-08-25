# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import time
import numpy as np
import utils.plackettluce as pl
from utils.sample_group_assignment import GDL23
import utils.ranking as rnk


def PL_rank_1(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
    n_docs = labels.shape[0]
    result = np.zeros(n_docs, dtype=np.float64)
    cutoff = min(rank_weights.shape[0], n_docs)

    if n_docs == 1:
        return np.zeros_like(scores)

    assert n_samples is not None or sampled_rankings is not None
    if sampled_rankings is None:
        sampled_rankings = pl.gumbel_sample_rankings(scores, n_samples, cutoff=cutoff)[
            0
        ]
    else:
        n_samples = sampled_rankings.shape[0]

    srange = np.arange(n_samples)
    crange = np.arange(cutoff)

    weighted_labels = labels[sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    np.add.at(result, sampled_rankings, cumsum_labels)
    result /= n_samples

    placed_mask = np.zeros((n_samples, cutoff - 1, n_docs), dtype=np.bool)
    placed_mask[srange[:, None], crange[None, :-1], sampled_rankings[:, :-1]] = True
    placed_mask[:, :] = np.cumsum(placed_mask, axis=1)

    total_denom = np.logaddexp.reduce(scores)
    minus_denom = np.logaddexp.accumulate(scores[sampled_rankings[:, :-1]], axis=1)
    denom_per_rank = np.log(1.0 - np.exp(minus_denom - total_denom)) + total_denom
    prob_per_rank = np.empty((n_samples, cutoff, n_docs), dtype=np.float64)
    prob_per_rank[:, 0, :] = np.exp(scores[None, :] - total_denom)
    prob_per_rank[:, 1:, :] = np.exp(scores[None, None, :] - denom_per_rank[:, :, None])
    prob_per_rank[:, 1:, :][placed_mask] = 0.0

    minus_weights = np.mean(
        np.sum(prob_per_rank * cumsum_labels[:, :, None], axis=1),
        axis=0,
        dtype=np.float64,
    )

    result -= minus_weights

    return result


def PL_rank_2(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
    n_docs = labels.shape[0]
    result = np.zeros(n_docs, dtype=np.float64)
    cutoff = min(rank_weights.shape[0], n_docs)

    if n_docs == 1:
        return np.zeros_like(scores)

    assert n_samples is not None or sampled_rankings is not None
    if sampled_rankings is None:
        sampled_rankings = pl.gumbel_sample_rankings(scores, n_samples, cutoff=cutoff)[
            0
        ]
    else:
        n_samples = sampled_rankings.shape[0]

    srange = np.arange(n_samples)
    crange = np.arange(cutoff)

    relevant_docs = np.where(np.not_equal(labels, 0))[0]
    n_relevant_docs = relevant_docs.size

    weighted_labels = labels[sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    np.add.at(result, sampled_rankings[:, :-1], cumsum_labels[:, 1:])
    result /= n_samples

    placed_mask = np.zeros((n_samples, cutoff - 1, n_docs), dtype=np.bool)
    placed_mask[srange[:, None], crange[None, :-1], sampled_rankings[:, :-1]] = True
    placed_mask[:, :] = np.cumsum(placed_mask, axis=1)

    total_denom = np.logaddexp.reduce(scores)
    minus_denom = np.logaddexp.accumulate(scores[sampled_rankings[:, :-1]], axis=1)
    denom_per_rank = (
        np.log(np.maximum(1.0 - np.exp(minus_denom - total_denom), 10**-8))
        + total_denom
    )
    prob_per_rank = np.empty((n_samples, cutoff, n_docs), dtype=np.float64)
    prob_per_rank[:, 0, :] = np.exp(scores[None, :] - total_denom)
    prob_per_rank[:, 1:, :] = np.exp(scores[None, None, :] - denom_per_rank[:, :, None])
    prob_per_rank[:, 1:, :][placed_mask] = 0.0

    result -= np.mean(
        np.sum(prob_per_rank * cumsum_labels[:, :, None], axis=1),
        axis=0,
        dtype=np.float64,
    )
    result[relevant_docs] += np.mean(
        np.sum(
            prob_per_rank[:, :, relevant_docs]
            * (rank_weights[None, :cutoff, None] * labels[None, None, relevant_docs]),
            axis=1,
        ),
        axis=0,
        dtype=np.float64,
    )

    return result


def PL_rank_3(rank_weights, labels, scores, n_samples=None, sampled_rankings=None):
    n_docs = labels.shape[0]
    result = np.zeros(n_docs, dtype=np.float64)
    cutoff = min(rank_weights.shape[0], n_docs)

    if n_docs == 1:
        return np.zeros_like(scores)

    scores = scores.copy() - np.amax(scores) + 10.0

    assert n_samples is not None or sampled_rankings is not None
    if sampled_rankings is None:
        sampled_rankings = pl.gumbel_sample_rankings(
            scores,
            n_samples,
            cutoff=cutoff,
            group_ids=np.arange(n_docs),
            return_full_rankings=True,
        )[0]
    else:
        n_samples = sampled_rankings.shape[0]

    cutoff_sampled_rankings = sampled_rankings[:, :cutoff]

    srange = np.arange(n_samples)

    relevant_docs = np.where(np.not_equal(labels, 0))[0]
    n_relevant_docs = relevant_docs.size

    weighted_labels = labels[cutoff_sampled_rankings] * rank_weights[None, :cutoff]
    cumsum_labels = np.cumsum(weighted_labels[:, ::-1], axis=1)[:, ::-1]

    np.add.at(result, cutoff_sampled_rankings[:, :-1], cumsum_labels[:, 1:])
    result /= n_samples

    exp_scores = np.exp(scores).astype(np.float64)
    denom_per_rank = np.cumsum(exp_scores[sampled_rankings[:, ::-1]], axis=1)[
        :, : -cutoff - 1 : -1
    ]

    cumsum_weight_denom = np.cumsum(rank_weights[:cutoff] / denom_per_rank, axis=1)
    cumsum_reward_denom = np.cumsum(cumsum_labels / denom_per_rank, axis=1)

    if cutoff < n_docs:
        second_part = -exp_scores[None, :] * cumsum_reward_denom[:, -1, None]
        second_part[:, relevant_docs] += (
            labels[relevant_docs][None, :]
            * exp_scores[None, relevant_docs]
            * cumsum_weight_denom[:, -1, None]
        )
    else:
        second_part = np.empty((n_samples, n_docs), dtype=np.float64)

    sampled_direct_reward = (
        labels[cutoff_sampled_rankings]
        * exp_scores[cutoff_sampled_rankings]
        * cumsum_weight_denom
    )
    sampled_following_reward = exp_scores[cutoff_sampled_rankings] * cumsum_reward_denom
    second_part[srange[:, None], cutoff_sampled_rankings] = (
        sampled_direct_reward - sampled_following_reward
    )

    return result + np.mean(second_part, axis=0)


def Group_Fair_PL(
    rank_weights,
    labels,
    group_membership,
    fairness_constraints,
    scores,
    n_samples=None,
    group_n_samples=1,
    sampled_rankings=None,
):
    n_docs = labels.shape[0]
    # TODO: check what the following if block is for
    if n_docs == 1:
        return np.zeros_like(scores)
    # sample a random group-fair group assigment from the fairness constraints
    n_groups = len(fairness_constraints)
    n_docs = labels.shape[0]

    cutoff = min(rank_weights.shape[0], n_docs)

    # extract the gropu-wise scores and labels
    group_scores = np.zeros((n_groups, n_docs))
    group_rel_labels = np.zeros((n_groups, n_docs))
    group_ids = []
    for j in range(n_groups):
        # TODO: check group_membership is a numpy array
        group_ids.append(np.where(group_membership == j)[0])
        group_scores[j][group_ids[j]] = scores[group_ids[j]]
        group_rel_labels[j][group_ids[j]] = labels[group_ids[j]]

    # Remember to return numpy array. Otherwise the np.where function does not work.
    x_sampler = GDL23(n_groups, cutoff, fairness_constraints)
    sampled_group_assignments = np.asarray(x_sampler.sample(n_samples))

    before = time.time()
    sampled_rankings = []
    result = np.zeros(n_docs, dtype=np.float64)
    for x in sampled_group_assignments:
        group_n_samples = 1
        for j in range(n_groups):
            group_result = np.zeros(n_docs)
            # don't have to normalize the scores as they are already between 0 and 1.
            # group_scores[j] = group_scores[j].copy() - np.amax(group_scores[j]) + 10.
            group_ranks = np.where(x == j)[0]
            if (
                len(group_ranks) == 0
            ):  # if none of the ranks were assigned to this group
                continue
            group_cutoff = len(group_ranks)
            group_sampled_rankings = pl.gumbel_sample_rankings(
                group_scores[j],
                group_n_samples,
                cutoff=group_cutoff,
                group_ids=group_ids[j],
                return_full_rankings=True,
            )[0]

            # adjust the following code for group-wise computation
            cutoff_sampled_rankings = group_sampled_rankings[:, :group_cutoff]

            srange = np.arange(group_n_samples)

            relevant_docs = np.where(np.not_equal(group_rel_labels[j], 0))[0]
            n_relevant_docs = relevant_docs.size

            group_rank_weights = rank_weights[group_ranks]
            weighted_group_rel_labels = (
                group_rel_labels[j][cutoff_sampled_rankings]
                * group_rank_weights[None, :group_cutoff]
            )
            cumsum_labels = np.cumsum(weighted_group_rel_labels[:, ::-1], axis=1)[
                :, ::-1
            ]

            np.add.at(
                group_result, cutoff_sampled_rankings[:, :-1], cumsum_labels[:, 1:]
            )
            group_result /= group_n_samples

            # group_exp_scores = np.array([np.exp(y) for y in group_scores[j] if y > 0]).astype(np.float64)
            group_exp_scores = group_scores[j].copy()
            group_exp_scores[group_exp_scores > 0] = np.exp(
                group_exp_scores[group_exp_scores > 0]
            )
            denom_per_rank = np.cumsum(
                group_exp_scores[group_sampled_rankings[:, ::-1]], axis=1
            )[:, : -group_cutoff - 1 : -1]

            cumsum_weight_denom = np.cumsum(
                group_rank_weights[:group_cutoff] / denom_per_rank, axis=1
            )
            cumsum_reward_denom = np.cumsum(cumsum_labels / denom_per_rank, axis=1)

            if group_cutoff < n_docs:
                second_part = (
                    -group_exp_scores[None, :] * cumsum_reward_denom[:, -1, None]
                )
                second_part[:, relevant_docs] += (
                    group_rel_labels[j][relevant_docs][None, :]
                    * group_exp_scores[None, relevant_docs]
                    * cumsum_weight_denom[:, -1, None]
                )
            else:
                second_part = np.empty((group_n_samples, n_docs), dtype=np.float64)

            sampled_direct_reward = (
                group_rel_labels[j][cutoff_sampled_rankings]
                * group_exp_scores[cutoff_sampled_rankings]
                * cumsum_weight_denom
            )
            sampled_following_reward = (
                group_exp_scores[cutoff_sampled_rankings] * cumsum_reward_denom
            )
            second_part[srange[:, None], cutoff_sampled_rankings] = (
                sampled_direct_reward - sampled_following_reward
            )

            group_result += np.mean(second_part, axis=0)
            result += group_result
    print("Time without multiprocessing: ", time.time() - before)
    return result / n_samples
