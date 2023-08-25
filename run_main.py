# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import datetime as dt
import json
import os
import pathlib
import time

import numpy as np
import tensorflow as tf

import algorithms.PLRank_multiprocessing as plr
import algorithms.stochasticrank as sr
import algorithms.tensorflowloss as tfl
import utils.dataset as dataset
import utils.evaluate as evl
import utils.nnmodel as nn
from read_args import read_args

mytime = dt.datetime.now()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

args, config = read_args()
num_samples = config["num_samples"]

if num_samples == "dynamic":
    dynamic_samples = True
else:
    dynamic_samples = False
    num_samples = int(num_samples)

p = pathlib.Path("local_output/")
p.mkdir(parents=True, exist_ok=True)


if args.bias > -1:
    dataset_name = config["dataset"] + "biased_0_" + str(args.bias)
else:
    dataset_name = config["dataset"]


def read_dataset(config):
    print("Reading dataset...")
    data = dataset.get_dataset_from_json_info(
        dataset_name=dataset_name,
        info_path=config["dataset_info_path"],
        read_from_pickle=False,
    )
    fold_id = (config["fold_id"] - 1) % data.num_folds()
    data = data.get_data_folds()[fold_id]
    data.read_data()
    return data


output_path = (
    "./local_output/"
    + dataset_name
    + "/"
    + str(mytime)
    + "_loss="
    + args.loss
    + "_k="
    + str(config["cutoff"])
    + "_fairness="
    + config["fairness_requirement"]
    + "_nsamples="
    + str(config["num_samples"])
    + "_delta="
    + str(config["delta"])
    + "_run="
    + str(args.run_no)
    + ".json"
)


isExist = os.path.exists("./local_output/" + dataset_name + "/")
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs("./local_output/" + dataset_name + "/")

data = read_dataset(config)
n_queries = data.train.num_queries()


epoch_results = []
timed_results = []

max_ranking_size = np.min((config["cutoff"], data.max_query_size()))
metric_weights = 1.0 / np.log2(np.arange(max_ranking_size + 1) + 2)[:max_ranking_size]
train_labels = 2**data.train.label_vector - 1
valid_labels = 2**data.validation.label_vector - 1
test_labels = 2**data.test.label_vector - 1

train_gender_labels = data.train.gender_vector
valid_gender_labels = data.validation.gender_vector
test_gender_labels = data.test.gender_vector
n_groups = np.maximum(
    len(np.unique(train_gender_labels)), len(np.unique(valid_gender_labels))
)

fairness_constraints = []
prefix_fairness_constraints = []
proportions = np.array(
    [
        (
            len(np.where(train_gender_labels == j)[0])
            + len(np.where(valid_gender_labels == j)[0])
        )
        / (len(train_gender_labels) + len(valid_gender_labels))
        for j in range(n_groups)
    ]
)

if config["fairness_requirement"] == "Equal":
    for t in range(config["cutoff"]):
        lower = []
        upper = []
        for j in range(n_groups):
            l = t * (1 / n_groups - config["delta"])
            u = t * (1 / n_groups + config["delta"])
            lower.append(l)
            upper.append(u)
        prefix_fairness_constraints.append([lower, upper])
    fairness_constraints.append(
        [
            int(np.floor(config["cutoff"] * (1 / n_groups - config["delta"])))
            for _ in range(n_groups)
        ]
    )  # lower bounds
    fairness_constraints.append(
        [
            int(np.ceil(config["cutoff"] * (1 / n_groups + config["delta"])))
            for _ in range(n_groups)
        ]
    )  # upper bounds
elif config["fairness_requirement"] == "Proportional":
    for t in range(config["cutoff"]):
        lower = []
        upper = []
        for j in range(n_groups):
            l = t * (proportions[j] - config["delta"])
            u = t * (proportions[j] + config["delta"])
            lower.append(l)
            upper.append(l + 1)
        prefix_fairness_constraints.append([lower, upper])
    fairness_constraints.append(
        [
            int(np.floor(config["cutoff"] * (proportions[j] - config["delta"])))
            for j in range(n_groups)
        ]
    )
    fairness_constraints.append(
        [
            int(np.ceil(config["cutoff"] * (proportions[j] + config["delta"])))
            for j in range(n_groups)
        ]
    )

fairness_constraints = np.asarray(fairness_constraints, dtype=np.int32)
prefix_fairness_constraints = np.asarray(prefix_fairness_constraints, dtype=np.int32)

if config["verbose"]:
    print(proportions, fairness_constraints)
ideal_train_metrics = evl.ideal_metrics(data.train, metric_weights, train_labels)
ideal_valid_metrics = evl.ideal_metrics(data.validation, metric_weights, valid_labels)
ideal_test_metrics = evl.ideal_metrics(data.test, metric_weights, test_labels)


model_params = {
    "hidden units": [32, 32],
    "learning_rate": config["learning_rate"],
    "learning_rate_decay": 1.0,
}
model = nn.init_model(model_params)

postprocess_algorithms = args.postprocess_algorithms.split(",")

# print("Before valid: ", time.time())
# valid_result = evl.compute_results(
#     data.validation,
#     model,
#     metric_weights,
#     valid_labels,
#     ideal_valid_metrics,
#     config["num_eval_samples"],
#     valid_gender_labels,
#     fairness_constraints,
#     prefix_fairness_constraints,
#     postprocess_algorithms=postprocess_algorithms,
# )
# print("After valid: ", time.time())
# test_result = evl.compute_results(
#     data.test,
#     model,
#     metric_weights,
#     test_labels,
#     ideal_test_metrics,
#     config["num_eval_samples"],
#     test_gender_labels,
#     fairness_constraints,
#     prefix_fairness_constraints,
#     postprocess_algorithms=postprocess_algorithms,
# )
# print("After test: ", time.time())

real_start_time = time.time()
total_train_time = 0
last_total_train_time = time.time()
method_train_time = 0


if num_samples == "dynamic":
    dynamic_samples = True
    float_num_samples = 10.0
    add_per_step = 90.0 / (n_queries * 40.0)
    max_num_samples = 1000

# time_points = np.arange(0, config["maximum_train_time"], 1).astype(np.float64)

# time_i = 1
# n_times = time_points.shape[0]

steps = 0
batch_size = config["batch_size"]
optimizer = tf.keras.optimizers.SGD(
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=model_params["learning_rate"],
        decay_steps=n_queries / batch_size,
        decay_rate=model_params["learning_rate_decay"],
    )
)

epoch_i = -1
is_last_epoch = False
# each epoch is an iteration over all queries
while epoch_i < config["n_epochs"]:
    if config["verbose"]:
        print("EPOCH: %04d.00 TIME: %04d" % (epoch_i, total_train_time))
    epoch_i += 1
    if dynamic_samples:
        num_samples = int(max(1, np.ceil(200 * np.sqrt(epoch_i))))

    query_permutation = np.random.permutation(n_queries)
    # iterate over batch of queries
    for batch_i in range(int(np.ceil(n_queries / batch_size))):
        batch_queries = query_permutation[
            batch_i * batch_size : (batch_i + 1) * batch_size
        ]
        cur_batch_size = batch_queries.shape[0]

        batch_features = []
        # get document features for each query
        for i in range(cur_batch_size):
            batch_features.append(data.train.query_feat(batch_queries[i]))

        # calculate cumulative number of documents for each query in this batch
        batch_ranges = np.zeros(cur_batch_size + 1, dtype=np.int32)
        batch_ranges[1:] = [
            batch_features[idx].shape[0] for idx in range(cur_batch_size)
        ]
        batch_ranges = np.cumsum(batch_ranges)

        # concatenate all document features into a single matrix
        batch_features = np.concatenate(batch_features, axis=0)

        with tf.GradientTape() as tape:
            batch_tf_scores = model(batch_features)
            # tf.debugging.assert_non_negative(batch_tf_scores, message="Some scores are negative")
            loss = 0
            batch_doc_weights = np.zeros(batch_features.shape[0], dtype=np.float64)
            use_doc_weights = False
            for i, qid in enumerate(batch_queries):
                q_labels = data.train.query_values_from_vector(qid, train_labels)
                q_gender_labels = data.train.query_values_from_vector(
                    qid, train_gender_labels
                )
                q_feat = batch_features[batch_ranges[i] : batch_ranges[i + 1], :]
                q_ideal_metric = ideal_train_metrics[qid]

                if q_ideal_metric != 0:
                    q_metric_weights = (
                        metric_weights  # /q_ideal_metric #uncomment for NDCG
                    )

                    q_tf_scores = batch_tf_scores[batch_ranges[i] : batch_ranges[i + 1]]

                    last_method_train_time = time.time()
                    if args.loss == "policygradient":
                        loss += tfl.policy_gradient(
                            q_metric_weights,
                            q_labels,
                            q_tf_scores,
                            n_samples=num_samples,
                        )

                    elif args.loss == "placementpolicygradient":
                        loss += tfl.placement_policy_gradient(
                            q_metric_weights,
                            q_labels,
                            q_tf_scores,
                            n_samples=num_samples,
                        )

                    else:
                        q_np_scores = q_tf_scores.numpy()[:, 0]

                        if args.loss == "Group-Fair-PL":
                            doc_weights = plr.Group_Fair_PL(
                                q_metric_weights,
                                q_labels,
                                q_gender_labels,
                                fairness_constraints,
                                q_np_scores,
                                n_samples=num_samples,
                                group_n_samples=1,
                            )
                        elif args.loss == "PL-rank-3":
                            doc_weights = plr.PL_rank_3(
                                q_metric_weights,
                                q_labels,
                                q_np_scores,
                                n_samples=num_samples,
                            )
                        else:
                            raise NotImplementedError("Unknown loss %s" % args.loss)

                        batch_doc_weights[
                            batch_ranges[i] : batch_ranges[i + 1]
                        ] = doc_weights
                        use_doc_weights = True
                    method_train_time += time.time() - last_method_train_time

            if use_doc_weights:
                loss = -tf.reduce_sum(batch_tf_scores[:, 0] * batch_doc_weights)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        steps += cur_batch_size
        if dynamic_samples:
            float_num_samples = 10 + steps * add_per_step
            num_samples = min(int(np.round(float_num_samples)), max_num_samples)

    if epoch_i % 50 == 0:
        if epoch_i == config["n_epochs"]:
            is_last_epoch = True

        total_train_time += time.time() - last_total_train_time
        print("before valid: ", time.time())
        valid_result = evl.compute_results(
            data.validation,
            model,
            metric_weights,
            valid_labels,
            ideal_valid_metrics,
            config["num_eval_samples"],
            valid_gender_labels,
            fairness_constraints,
            prefix_fairness_constraints,
            postprocess_algorithms,
            is_last_epoch,
        )
        print("after valid: ", time.time())
        test_result = evl.compute_results(
            data.test,
            model,
            metric_weights,
            test_labels,
            ideal_test_metrics,
            config["num_eval_samples"],
            test_gender_labels,
            fairness_constraints,
            prefix_fairness_constraints,
            postprocess_algorithms,
            is_last_epoch,
        )
        print("after test: ", time.time())
        print(
            "EPOCH: %07.2f TIME: %04d"
            " VALI: exp: %0.4f"
            " TEST: exp: %0.4f"
            % (
                epoch_i,
                method_train_time,
                valid_result[0]["normalized expectation"],
                test_result[0]["normalized expectation"],
            )
        )

        cur_result = {
            "steps": steps,
            "epoch": epoch_i,
            "train time": method_train_time,
            "total time": total_train_time,
            "validation result": valid_result,
            "test result": test_result,
            "num_samples": num_samples,
        }
        epoch_results.append(cur_result)
        last_total_train_time = time.time()

output = {
    "dataset": dataset_name,
    "fold number": config["fold_id"],
    "run name": args.loss.replace("_", " "),
    "loss": args.loss.replace("_", " "),
    "model hyperparameters": model_params,
    "epoch results": epoch_results,
    "number of samples": num_samples,
    "number of evaluation samples": config["num_eval_samples"],
    "cutoff": config["cutoff"],
    "fairness constraints": fairness_constraints.tolist(),
    "prefix fairness constraints": prefix_fairness_constraints.tolist(),
}
if dynamic_samples:
    output["number of samples"] = "dynamic"


print("Writing results to %s" % output_path)
with open(output_path, "w") as f:
    json.dump(output, f)
