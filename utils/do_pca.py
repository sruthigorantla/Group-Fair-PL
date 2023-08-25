import numpy as np
import argparse
import tqdm, os
from sklearn.decomposition import PCA
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="HMDA")
parser.add_argument("--n_components", type=int, default=50)
args = parser.parse_args()

if args.dataset == "HMDA":
    prot_attr = 130
else:
    raise ValueError("dataset not supported")

input_folder_path = "../data/" + args.dataset + "/"
output_folder_path = "../data/pca" + args.dataset + "/"
if os.path.exists(output_folder_path) == False:
    os.mkdir(output_folder_path)
n_components = args.n_components

# write code to read from train.txt, valid.txt, test.txt from ../data/HMDA directory
# and store the data in a numpy array X
X = {}
qid = {}
rel_label = {}
prot_label = {}
for split in ["train", "vali", "test"]:
    X[split] = []
    qid[split] = []
    rel_label[split] = []
    prot_label[split] = []
    print(input_folder_path + split + ".txt")
    for line in open(input_folder_path + split + ".txt", "r"):
        info = line[: line.find("#")].split()
        rel_label[split].append("rel:" + info[0])
        qid[split].append(info[1])
        prot_label_old = info[prot_attr].split(":")[1]
        prot_label[split].append("1:" + prot_label_old)
        feat_pairs = info[2:]
        del feat_pairs[prot_attr - 2]  # do not include group information in PCA
        doc_feat = []
        for pair in feat_pairs:
            _, feature = pair.split(":")
            feat_value = float(feature)
            doc_feat.append(feat_value)
        X[split].append(doc_feat)

pca = PCA(n_components=min(n_components, len(X["train"][0])))
pca.fit(X["train"])
pca_train = pca.transform(X["train"])
normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(pca_train)
newX = {}
for split in ["train", "vali", "test"]:
    with open(output_folder_path + split + ".txt", "w") as f:
        if split == "train":
            newX[split] = normalizer.transform(pca_train)
        else:
            pcaX = pca.transform(X[split])
            newX[split] = normalizer.transform(pcaX)
        for i in range(len(qid[split])):
            new_line = (
                rel_label[split][i]
                + " "
                + qid[split][i]
                + " "
                + prot_label[split][i]
                + " "
            )
            for j in range(min(n_components, len(newX["train"][0])) - 1):
                new_line += str(j + 2) + ":" + str(newX[split][i][j]) + " "
            new_line += str(j + 3) + ":" + str(newX[split][i][j + 1]) + "\n"

            f.write(new_line)
