import itertools
from collections import defaultdict
from typing import List

import numpy as np
from tqdm import tqdm
import math
import copy


class GDL23:
    def __init__(
        self, num_groups: int, rank_size: int, fairness_constraints: List[List[int]]
    ):
        assert num_groups >= 0
        assert rank_size >= 0
        self.num_groups = num_groups
        self.rank_size = rank_size

        # Add dummy entry to bounds to ensure that can
        # be indexed as per group id
        self.lower_bounds = np.concatenate(([0], fairness_constraints[0]))
        self.upper_bounds = np.concatenate(([0], fairness_constraints[1]))

        self.lower_cumsum = np.cumsum(self.lower_bounds)

        self._check_bounds()
        self.create_dp_table()
        # self.intra_group_ranking = self.get_intra_group_ranking()

    def get_intra_group_ranking(self):
        intra_group_ranking = defaultdict(list)

        counter = np.zeros(self.num_groups)
        for item in self.true_ranking:
            intra_group_ranking[self.id_2_group[item]].append(item)
            counter[self.id_2_group[item]] += 1
            if (counter > self.rank_size).all():
                break

        return intra_group_ranking

    def get_bounds(self, proportions, delta):
        L_k = []
        U_k = []
        for j in range(self.num_groups):
            L_k.append(max(0, math.ceil((proportions[j] - delta) * self.rank_size)))
            U_k.append(math.floor((proportions[j] + delta) * self.rank_size))

        return L_k, U_k

    def _check_bounds(
        self,
    ):
        assert sum(self.lower_bounds[1:]) <= self.rank_size
        assert sum(self.upper_bounds[1:]) >= self.rank_size
        assert not any(
            [a > b for (a, b) in zip(self.lower_bounds[1:], self.upper_bounds[1:])]
        )

    def create_dp_table(
        self,
    ):
        # Create a dp table
        self.dp = dp = np.zeros((self.rank_size + 1, self.num_groups + 1))

        # Base case of dp table
        for i in range(self.num_groups + 1):
            dp[0, i] = 1 if self.lower_cumsum[i] <= 0 else 0

        # Loop over each rank size
        for k in range(1, self.rank_size + 1):
            # For a rank size, loop over groups
            for i in range(1, self.num_groups + 1):
                sum = 0
                for x in range(self.lower_bounds[i], self.upper_bounds[i] + 1):
                    if (k - x) >= 0:
                        sum += dp[k - x, i - 1]
                dp[k, i] = sum

    # def construct_ranking(self, answer):

    #     final_ranking = []

    #     intra_group_ranking = copy.deepcopy(self.intra_group_ranking)

    #     for item in answer:
    #         final_ranking.append(intra_group_ranking[item].pop(0))

    #     return final_ranking

    def sample(self, num_samples):
        final_rankings = []
        sample_count = 0
        while sample_count < num_samples:
            sample_count += 1
            sum = self.rank_size
            i = self.num_groups
            answer = []

            while sum > 0:
                # construct probability distribution for x_i in i_th group
                distribution = [0] * (self.upper_bounds[i] - self.lower_bounds[i] + 1)

                for idx, x in enumerate(
                    range(self.lower_bounds[i], self.upper_bounds[i] + 1)
                ):
                    if sum - x >= 0:
                        distribution[idx] = self.dp[sum - x, i - 1] / self.dp[sum, i]
                    else:
                        distribution[idx] = 0

                # Ensure that it is a prob distribution
                assert np.isclose(np.sum(distribution), 1.0)

                # Sample from the categorical distribution
                rng = np.random.default_rng()
                sampled = rng.multinomial(1, distribution, size=1)
                x_sampled = np.argmax(sampled, axis=-1)[0]

                answer.append(self.lower_bounds[i] + x_sampled)

                # Search the count over other groups
                sum -= self.lower_bounds[i] + x_sampled
                i -= 1

            assert np.sum(answer) == self.rank_size
            group_counts = answer[::-1]
            while len(group_counts) < self.num_groups:
                group_counts = [0] + group_counts

            permutation = []
            for group in range(self.num_groups):
                permutation += [group] * group_counts[group]

            np.random.shuffle(permutation)

            # No need to map the group-wise ranking to  document ids
            # final_rankings.append(self.construct_ranking(permutation))
            final_rankings.append(permutation)

        return final_rankings


def manual_table(
    num_groups: int,
    rank_size: int,
    lower_bounds: List[int],
    upper_bounds: List[int],
):
    somelists = []
    for i in range(num_groups):
        somelists.append([x for x in range(lower_bounds[i], upper_bounds[i] + 1)])
    cnt = defaultdict(lambda: 0)
    for element in itertools.product(*somelists):
        cnt[sum(element)] += 1

    # for k in range(rank_size + 1):
    #     print(cnt[k])

    return cnt[rank_size]


if __name__ == "__main__":
    rank_size = 10
    num_groups = 4
    lower_bounds = [1, 0, 3, 2]
    upper_bounds = [5, 5, 5, 5]

    counts_table = GDL23(
        num_groups=num_groups,
        rank_size=rank_size,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    print(counts_table.dp)

    # sample of count tuple:
    sample = counts_table.sample()
    print(f"Sampled count tuple: {sample}")

    # Get empirical probabilities over valid count tuples
    # sampled from the DP sampler

    NUM_OF_TRIALS = 100000
    empirical_cnt = defaultdict(lambda: 0)
    samples = []
    for _ in tqdm(range(NUM_OF_TRIALS), total=NUM_OF_TRIALS):
        sample = counts_table.sample()
        empirical_cnt[tuple(sample)] += 1
        samples.append(tuple(sample))

    # Get the number of valid tuples by performing an exhaustive search
    # over the tuples to get the count of unique tuples that sum to rank_size (k)
    num_of_examples = manual_table(
        num_groups=num_groups,
        rank_size=rank_size,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    print(f"Num of valid examples: {num_of_examples}")

    print(f"Num of unique samples: {len(set(samples))}")

    empirical_distribution = []
    for key, val in empirical_cnt.items():
        empirical_distribution.append(val / NUM_OF_TRIALS)

    print()
    print(f"Uniform distribution prob (1/num_of_valid_tuples): {1 / num_of_examples}")

    print()
    print("Empirical Distribution: ")
    print(empirical_distribution)
