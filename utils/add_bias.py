import numpy as np
import argparse
import os, shutil

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="HMDA")
parser.add_argument("--bias", type=int, default=5)
parser.add_argument("--ispca", type=bool, default=False)
args = parser.parse_args()

if args.ispca:
    input_folder = "../data/pca" + args.dataset + "/"
    output_folder = "../data/pca" + args.dataset + "biased_0_" + str(args.bias) + "/"
else:
    input_folder = "../data/" + args.dataset + "/"
    output_folder = "../data/" + args.dataset + "biased_0_" + str(args.bias) + "/"
if os.path.exists(output_folder) == False:
    os.mkdir(output_folder)
print(args.bias)
for split in ["train", "vali"]:
    arr = np.loadtxt(input_folder + split + ".txt", delimiter=" ", dtype=str)
    for i in range(len(arr)):
        if args.dataset == "MOVLENS":
            gender = float(arr[i, 52].split(":")[1])
            if gender == 1.0:
                arr[i, 0] = "rel:" + str(
                    float("0." + str(args.bias)) * float(arr[i, 0].split(":")[1])
                )
            elif gender == 2.0:
                arr[i, 0] = "rel:" + str(
                    float("0." + str(args.bias)) * 0.95 * float(arr[i, 0].split(":")[1])
                )
            elif gender == 3.0:
                arr[i, 0] = "rel:" + str(
                    float("0." + str(args.bias)) * 0.9 * float(arr[i, 0].split(":")[1])
                )
            elif gender == 4.0:
                arr[i, 0] = "rel:" + str(
                    float("0." + str(args.bias)) * 0.85 * float(arr[i, 0].split(":")[1])
                )

        elif args.dataset == "German":
            gender = float(arr[i, 5].split(":")[1])
            if gender == 1.0:
                arr[i, 0] = "rel:" + str(
                    float("0." + str(args.bias)) * float(arr[i, 0].split(":")[1])
                )

        else:
            gender = float(arr[i, 2].split(":")[1])
            if gender == 1.0:
                arr[i, 0] = "rel:" + str(
                    float("0." + str(args.bias)) * float(arr[i, 0].split(":")[1])
                )

    with open(
        output_folder + split + ".txt",
        "w",
    ) as f:
        for i in range(len(arr)):
            f.write(" ".join(arr[i]) + "\n")

shutil.copy(input_folder + "test.txt", output_folder + "test.txt")
