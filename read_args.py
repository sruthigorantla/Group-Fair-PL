import argparse
import json5


def read_args():
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("-f", "--file", type=str, help="config file path")
    parser.add_argument("-l", "--loss", type=str, help="name of the loss function")
    parser.add_argument(
        "-p",
        "--postprocess_algorithms",
        type=str,
        help="whether to apply postprocessing",
        default="none,GDL23,GAK19",
    )
    parser.add_argument(
        "-b",
        "--bias",
        type=int,
        default=-1,
        help="bias of (0.bias) the dataset; use bias=-1 for no bias",
    )
    parser.add_argument("-r", "--run_no", type=int, help="run number")
    # Parse the command-line arguments
    args = parser.parse_args()

    # Read the JSONC config file
    with open(args.file, "r") as f:
        config = json5.loads(f.read())
        if config["verbose"]:
            print(config)

        return args, config
