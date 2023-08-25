from multiprocessing import Pool
import numpy as np

import utils.ranking as rnk
from utils.sample_group_assignment import GDL23
from algorithms.geyik import rank_attributes_geyik


def sample_rankings(
    log_scores, n_samples, cutoff=None, prob_per_rank=False, postprocess=None
):
    n_docs = log_scores.shape[0]
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    if prob_per_rank:
        rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

    log_scores = np.tile(log_scores[None, :], (n_samples, 1))
    rankings = np.empty((n_samples, ranking_len), dtype=np.int32)
    inv_rankings = np.empty((n_samples, n_docs), dtype=np.int32)
    rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)

    if cutoff:
        inv_rankings[:] = ranking_len

    for i in range(ranking_len):
        log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
        probs = np.exp(log_scores - log_denom[:, None])
        if prob_per_rank:
            rank_prob_matrix[i, :] = np.mean(probs, axis=0)
        cumprobs = np.cumsum(probs, axis=1)
        random_values = np.random.uniform(size=n_samples)
        greater_equal_mask = np.greater_equal(random_values[:, None], cumprobs)
        sampled_ind = np.sum(greater_equal_mask, axis=1)

        rankings[:, i] = sampled_ind
        inv_rankings[ind, sampled_ind] = i
        rankings_prob[:, i] = probs[ind, sampled_ind]
        log_scores[ind, sampled_ind] = np.NINF

    if prob_per_rank:
        return rankings, inv_rankings, rankings_prob, rank_prob_matrix
    else:
        return rankings, inv_rankings, rankings_prob


def gumbel_sample_rankings(
    log_scores,
    n_samples,
    cutoff=None,
    group_ids=None,
    inverted=False,
    doc_prob=False,
    prob_per_rank=False,
    return_gumbel=False,
    return_full_rankings=False,
):
    n_docs = group_ids.shape[0]
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    if prob_per_rank:
        rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

    gumbel_scores = np.zeros((n_samples, n_docs), dtype=np.float64)
    gumbel_samples = np.random.gumbel(size=(n_samples, len(group_ids)))
    gumbel_scores = log_scores[None, group_ids] + gumbel_samples

    rankings, inv_rankings = rnk.multiple_cutoff_rankings(
        -gumbel_scores,
        group_ids,
        ranking_len,
        invert=inverted,
        return_full_rankings=return_full_rankings,
    )

    if not doc_prob:
        if not return_gumbel:
            return rankings, inv_rankings, None, None, None
        else:
            return rankings, inv_rankings, None, None, gumbel_scores

    log_scores = np.tile(log_scores[None, :], (n_samples, 1))
    rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
    for i in range(ranking_len):
        # log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
        if group_ids is not None:
            group_scores = log_scores.copy()
            # do exponentiaion only for the items from the group from which we sample
            group_scores[group_ids] = np.exp(group_scores[group_ids])
            log_denom = np.log(np.sum(group_scores, axis=1))
            probs = np.exp(log_scores - log_denom[:, None])
            if prob_per_rank:
                rank_prob_matrix[i, :] = np.mean(probs, axis=0)
            rankings_prob[:, i] = probs[ind, rankings[:, i]]
            log_scores[ind, rankings[:, i]] = np.NINF
        else:
            log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
            probs = np.exp(log_scores - log_denom[:, None])
            if prob_per_rank:
                rank_prob_matrix[i, :] = np.mean(probs, axis=0)
            rankings_prob[:, i] = probs[ind, rankings[:, i]]
            log_scores[ind, rankings[:, i]] = np.NINF

    if return_gumbel:
        gumbel_return_values = gumbel_scores
    else:
        gumbel_return_values = None

    if prob_per_rank:
        return (
            rankings,
            inv_rankings,
            rankings_prob,
            rank_prob_matrix,
            gumbel_return_values,
        )
    else:
        return (
            rankings,
            inv_rankings,
            rankings_prob,
            None,
            gumbel_return_values,
        )


def process_rankings(
    sampled_group_assignments,
    group_ids,
    group_log_scores,
    fairness_constraints,
):
    n_groups = len(group_log_scores)
    ranking_len = len(sampled_group_assignments[0])
    sampled_rankings = []
    for x in sampled_group_assignments:
        ranking = np.zeros(ranking_len, dtype=np.int32)
        group_n_samples = 1
        for j in range(n_groups):
            # don't have to normalize the scores as they are already between 0 and 1.
            # group_scores[j] = group_scores[j].copy() - np.amax(group_scores[j]) + 10.
            group_ranks = np.where(x == j)[0]
            if (
                len(group_ranks) == 0
            ):  # if none of the ranks were assigned to this group
                continue
            group_cutoff = min(len(group_ids[j]), len(group_ranks))
            group_ranks = group_ranks[:group_cutoff]

            group_sampled_rankings = gumbel_sample_rankings(
                group_log_scores[j],
                group_n_samples,
                cutoff=group_cutoff,
                group_ids=group_ids[j],
                return_full_rankings=True,
            )[0]
            ranking[group_ranks] = group_sampled_rankings[0][:group_cutoff]

        sampled_rankings.append(ranking)

    return sampled_rankings


def sample_from_GDL23(
    log_scores,
    n_samples,
    group_ids,
    fairness_constraints,
    cutoff=None,
    post_processor=None,
):
    n_docs = log_scores.shape[0]
    n_groups = fairness_constraints.shape[1]
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    # extract the gropu-wise scores and labels
    group_log_scores = np.zeros((n_groups, n_docs))
    for j in range(n_groups):
        group_log_scores[j][group_ids[j]] = log_scores[group_ids[j]]

    # Remember to return numpy array. Otherwise the np.where function does not work.
    x_sampler = GDL23(n_groups, cutoff, fairness_constraints)
    sampled_group_assignments = np.asarray(x_sampler.sample(n_samples))

    results = process_rankings(
        sampled_group_assignments,
        group_ids,
        group_log_scores,
        fairness_constraints,
    )

    return np.asarray(results)


def sample_from_GAK19(
    log_scores,
    n_samples,
    group_ids,
    prefix_fairness_constraints,
    cutoff=None,
    post_processor=None,
):
    # first rank all the documents using normal plackett luce and then rerank using Geyik et al.
    n_docs = log_scores.shape[0]
    sampled_rankings = gumbel_sample_rankings(
        log_scores, n_samples, cutoff=n_docs, group_ids=np.arange(n_docs)
    )[0]
    # sampled_rankings, _, _ = sample_rankings(log_scores, n_samples, cutoff=n_docs)

    n_groups = prefix_fairness_constraints[0].shape[1]

    group_fair_rankings = []
    for x in sampled_rankings:
        # extract the gropu-wise rankings
        ranks_of_groups = []
        ranks_till_cutoff_of_groups = []
        for j in range(n_groups):
            ranks_of_groups.append(np.where(np.in1d(x, group_ids[j]))[0])
            ranks_till_cutoff_of_groups.append(
                np.where(np.in1d(x[:cutoff], group_ids[j]))[0]
            )
        y = rank_attributes_geyik(
            x, ranks_of_groups, prefix_fairness_constraints, cutoff
        )
        group_fair_rankings.append(y)
        ranks_of_groups = []
        for j in range(n_groups):
            ranks_of_groups.append(np.where(np.in1d(y, group_ids[j]))[0])

    return np.asarray(group_fair_rankings)


def sample_group_fair_rankings(
    log_scores,
    n_samples,
    group_ids,
    fairness_constraints,
    prefix_fairness_constraints=None,
    cutoff=None,
    postprocess=None,
):
    if postprocess == "GDL23":
        return sample_from_GDL23(
            log_scores,
            n_samples,
            group_ids,
            fairness_constraints,
            cutoff,
            postprocess,
        )
    elif postprocess == "GAK19":
        return sample_from_GAK19(
            log_scores,
            n_samples,
            group_ids,
            prefix_fairness_constraints,
            cutoff,
            postprocess,
        )
    else:
        raise ValueError("Unknown postprocessing method: %s" % postprocess)


def normal_sample_rankings(
    log_scores,
    n_samples,
    cutoff=None,
    inverted=False,
    return_gumbel=False,
    return_full_rankings=False,
):
    n_docs = log_scores.shape[0]
    ind = np.arange(n_samples)

    if cutoff:
        ranking_len = min(n_docs, cutoff)
    else:
        ranking_len = n_docs

    normal_samples = np.random.normal(size=(n_samples, n_docs))
    normal_scores = log_scores[None, :] + normal_samples

    rankings, inv_rankings = rnk.multiple_cutoff_rankings(
        -normal_scores,
        ranking_len,
        invert=inverted,
        return_full_rankings=return_full_rankings,
    )
    if return_gumbel:
        return rankings, inv_rankings, normal_scores
    else:
        return rankings, inv_rankings, None


def metrics_based_on_samples(
    sampled_rankings,
    weight_per_rank,
    addition_per_rank,
    weight_per_doc,
):
    cutoff = sampled_rankings.shape[1]
    return np.sum(
        np.mean(
            weight_per_doc[sampled_rankings] * weight_per_rank[None, :cutoff],
            axis=0,
        )
        + addition_per_rank[:cutoff],
        axis=0,
    )


def datasplit_metrics(
    data_split,
    policy_scores,
    weight_per_rank,
    addition_per_rank,
    weight_per_doc,
    query_norm_factors=None,
    n_samples=1000,
):
    cutoff = weight_per_rank.shape[0]
    n_queries = data_split.num_queries()
    results = np.zeros(
        (n_queries, weight_per_rank.shape[1]),
    )
    for qid in range(n_queries):
        q_doc_weights = data_split.query_values_from_vector(qid, weight_per_doc)
        if not np.all(np.equal(q_doc_weights, 0.0)):
            q_policy_scores = data_split.query_values_from_vector(qid, policy_scores)
            sampled_rankings = gumbel_sample_rankings(
                q_policy_scores, n_samples, cutoff=cutoff
            )[0]
            results[qid] = metrics_based_on_samples(
                sampled_rankings,
                weight_per_rank,
                addition_per_rank,
                q_doc_weights[:, None],
            )
    if query_norm_factors is not None:
        results /= query_norm_factors

    return np.mean(results, axis=0)
