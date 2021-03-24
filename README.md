# P2HNNS

## Introduction

The paper for this package is accepted by SIGMOD 2021

```bath
Qiang Huang, Yifan Lei, and Anthony K. H. Tung. Point-to-Hyperplane Nearest Neighbor 
Search Beyond the Unit Hypersphere. In Proceedings of the 2021 ACM SIGMOD/PODS 
International Conference on Management of Data, 2021.
```

This package provides the implementations and experiments of `NH` and `FH` for Point-to-Hyperplane Nearest Neighbor Search (P2HNNS) in high-dimensional Euclidean spaces. We also implement three state-of-the-art hyperplane hashing schemes `EH`, `BH`, and `MH` and two heuristic linear scan methods Random-Scan and Sorted-Scan for comparison.

## Datasets and Queries

We use five real-life datasets `Yelp`, `Music-100`, `GloVe`, `Tiny-1M`, and `Msong` in the experiments. For each dataset, we generate 100 hyperplane queries for evaluations. 

The datasets and queries can be found by the follow link:
https://drive.google.com/drive/folders/1aBFV4feZcLnQkDR7tjC-Kj7g3MpfBqv7?usp=sharing

The statistics of datasets and queries are summarized as follows.

| Datasets  | #Data Objects | Dimensionality | #Queries | Data Size | Type   |
| --------- | ------------- | -------------- | -------- | --------- | ------ |
| Yelp      | 77,079        | 50             | 100      | 14.7 MB   | Rating |
| Music-100 | 1,000,000     | 100            | 100      | 381.5 MB  | Rating |
| GloVe     | 1,183,514     | 100            | 100      | 451.5 MB  | Text   |
| Tiny-1M   | 1,000,000     | 384            | 100      | 1.43 GB   | Image  |
| Msong     | 992,272       | 420            | 100      | 1.55 GB   | Audio  |

## Compilation

This source package requires ```g++-8``` with ```c++17``` support. Before the compilation, please check whether the `g++-8` is installed. If not, please install it first. We provide a way to install `g++-8` in Ubuntu 18.04 as follows:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-8
sudo apt-get install gcc-8 (optional)
```

To compile the c++ code, please type commands as follows:

```bash
cd methods
make
```

## Run Experiments

We provide the bash scripts to reproduce all of experiments reported in the SIGMOD 2021 paper.

#### Step 1: Get the Datasets and Generate Hyperplane Queries

Please download the [datasets](https://drive.google.com/drive/folders/1aBFV4feZcLnQkDR7tjC-Kj7g3MpfBqv7?usp=sharing) and copy them to the directory `data/bin/original/`. For example, when you get `Msong.bin`, please move it to `data/bin/original/Msong.bin`.

Once you finish copying the datasets to `data/bin/original/`, you can get the datasets and hyperplane queries (from `data/bin/` and `data/bin_normalized/`) with the following commands:

```bash
cd data/bin/original/
bash run.sh
```

#### Step 2: Run Experiments

You can reproduce all of the experiments with the following commands:

```bash
cd methods
bash run_all.sh
```

#### Step 3: Draw Figures

Finally, to reproduce all results, we also provide python scripts (i.e., `plot.py` and `plot_heatmap.py`) to draw the figures appeared in the SIGMOD 2021 paper. These two scripts require `python 3.7` (or higher verison) with numpy, scipy, and matplotlib installed. If you did not have `numpy, scipy, and matplotlib`, please first use `pip` install them before the next step.

Once everything is ready, you can type commands as follows to draw the figures with the experimental results.

```bash
python3 plot.py
python3 plot_heatmap.py
```

Please contact me (huangq@comp.nus.edu.sg) if you meet any issue. Thank you.