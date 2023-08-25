
## Previous works used in this paper
### 1. Learning-to-Rank at the Speed of Sampling: Plackett-Luce Gradient Estimation With Minimal Computational Complexity
[This repository](https://github.com/HarrieO/2022-SIGIR-plackett-luce) contains the code for PL-Rank-3; ([pdf available here](https://harrieo.github.io//publication/2022-sigir-short)).

### 2. Sampling Ex-Post Group-Fair Rankings
[This repository](https://github.com/sruthigorantla/sampling_random_group_fair_rankings) contains the code used for the distribution over the fair group assignments; ([pdf available here](https://arxiv.org/pdf/2203.00887.pdf)).


<span style="color:red">Both of the above algorithms are also included in our implementation. The users need not download anything from the repositories above to replicate our results.</span>


Our algorithm
-------

Necessary packages are in ``requirements.txt``

A file is required that explains the location and details of the LTR datasets available on the system, example file is available. Copy the file: ``local_dataset_info.txt``
Open this and edit the paths to the folders where the train/test/vali files are placed. 

Many experiments can be run one after the other by removing the comments in the file ``run.sh``
There are two kinds of input to ``run_main.py``:
- args:
  The command line arguements are read in the ``read_args.py`` script. It parses the following arguemnts.
  - "-f", "--file", type=str, help="config file path"
  - "-l", "--loss", type=str, help="name of the loss function"
  - "-p", "--postprocess_algorithms", type=str, help="whether to apply postprocessing", default="none,GDL23,GAK19",
  - "-b", "--bias", type=int, default=-1, help="bias of (0.bias) on the dataset; use bias=-1 for no bias",
  - "-r", "--run_no", type=int, help="run number"
- config:
  This is a config file in jsonc format useful to indicate hyperparameters of the experiments. Appropriate descriptions of the parameters are contained in the file. For example, see ``config_German.jsonc``.

The following command optimizes our relevance metric with Group-Fair-PL on the German Credit dataset:
```
python run_main.py --file config_German.jsonc --loss Group-Fair-PL --postprocess_algorithms none --run_no $i --bias $b
```

The following command optimizes NDCG with PL-rank-3 on the German Credit dataset with (1) no postproccessing, (2) GDL23 postprocessing and (3) GAK19 postprocessing algroithms:
```
python run_main.py --file config_German.jsonc --loss PL-rank-3 --postprocess_algorithms none,GDL23,GAK19 --run_no $i --bias $b
```
---
To add bias to the dataset, go to ``utils.py`` and run ``add_bias.py`` with a command line arguemnt ``--bias`` indicating the bias (the number after the decimal point in the bias factor). For example, the following command adds bias of 0.5 on the German credit dataset and creates a new data folder called Germanbiased_0_5
```
python add_bias.py --dataset German --bias 5 --ispca false
```
The ``--ispca`` indicates whether we want to use the pca preprocessed dataset. PCA preprocessing can be applied using ``do_pca.py``.

---
Example datasets have been included in this repository in ``data`` folder.