import random

import numpy as np


def rank_attributes_geyik_old(group_ids, group_proportions, groupwise_scores, cutoff):
    counts = {g_id: 0 for g_id in group_ids}
    ranked_docs = []
    ranked_scores = []
    ranked_groups = []

    for rank in range(1, cutoff + 1):
        belowMin = [
            g_id for g_id in group_ids if counts[g_id] < rank * group_proportions[g_id]
        ]
        belowMax = [
            g_id
            for g_id in group_ids
            if (
                (counts[g_id] >= rank * group_proportions[g_id])
                and (counts[g_id] < (rank + 1) * group_proportions[g_id])
            )
        ]

        groups_to_consider = None
        if len(belowMin) > 0:
            groups_to_consider = belowMin
        else:
            groups_to_consider = belowMax

        if groups_to_consider:
            # curr_data is a list of tuples of the form (group_id, doc_id, score)
            # where only the groups in groups_to_consider are considered
            # and the (doc, scores) are from the max available score for that group
            # which is referred to using the counts dictionary for that group
            curr_data = []
            for g_id in groups_to_consider:
                curr_data.append((g_id, *groupwise_scores[g_id][counts[g_id]]))

            # sort the curr_data in decreasing order of scores
            curr_data = sorted(curr_data, key=lambda x: x[2], reverse=True)

            # get the max score and its corresponding doc_id and group_id
            max_g_id, max_doc_id, max_score = curr_data[0]

            ranked_docs.append(max_doc_id)
            ranked_scores.append(max_score)
            ranked_groups.append(max_g_id)

            # increment the count for the group that was selected
            counts[max_g_id] += 1
        else:
            raise ValueError("No groups to consider")

    return ranked_groups, ranked_docs, ranked_scores


def rank_attributes_geyik(
    full_ranking, ranks_of_groups, prefix_fairness_constraints, cutoff
):
    counts = {g_id: 0 for g_id in np.arange(len(prefix_fairness_constraints[0][0]))}
    n_groups = len(counts.keys())
    ranked_docs = []

    for t in range(cutoff):
        belowMin = [
            g_id
            for g_id in np.arange(n_groups)
            if (counts[g_id] < prefix_fairness_constraints[t][0][g_id])
            and (
                counts[g_id] < len(ranks_of_groups[g_id])
            )  # fairness_constraints[0] is the lower bound
        ]
        belowMax = [
            g_id
            for g_id in np.arange(n_groups)
            if (
                (counts[g_id] >= prefix_fairness_constraints[t][0][g_id])
                and (counts[g_id] < prefix_fairness_constraints[t][1][g_id])
                and (counts[g_id] < len(ranks_of_groups[g_id]))
            )
        ]

        groups_to_consider = None
        if len(belowMin) > 0:
            groups_to_consider = belowMin
        else:
            groups_to_consider = belowMax

        if groups_to_consider:
            min_rank_group = groups_to_consider[
                np.argmin(
                    [ranks_of_groups[g_id][counts[g_id]] for g_id in groups_to_consider]
                )
            ]
            ranked_docs.append(
                full_ranking[ranks_of_groups[min_rank_group][counts[min_rank_group]]]
            )

            # increment the count for the group that was selected
            counts[min_rank_group] += 1
        else:
            # append the full ranking
            index = 0
            while len(ranked_docs) < cutoff:
                if full_ranking[index] not in ranked_docs:
                    ranked_docs.append(full_ranking[index])
                index += 1
            break

    return ranked_docs


def main():
    group_ids = [1, 2]
    num_groups = len(group_ids)
    num_total_docs = 1000
    cutoff = 20

    doc_scores = np.random.uniform(10, 20, size=num_total_docs)

    group_to_doc_ids = {
        1: np.arange(0, num_total_docs / 2, dtype=int),
        2: np.arange(num_total_docs / 2, num_total_docs, dtype=int),
    }
    group_proportions = {1: 0.5, 2: 0.5}

    arr = {}
    # generate random scores between 10 and 20 for each row and sort them in decreasing order
    for g_id in group_ids:
        groupwise_scores = [
            (doc_id, doc_scores[doc_id]) for doc_id in group_to_doc_ids[g_id]
        ]

        arr[g_id] = sorted(groupwise_scores, key=lambda x: x[1], reverse=True)

    (
        ranked_group_list,
        ranked_doc_list,
        ranked_score_list,
    ) = rank_attributes_geyik(group_ids, group_proportions, arr, cutoff)

    return ranked_group_list, ranked_doc_list, ranked_score_list


if __name__ == "__main__":
    main()
