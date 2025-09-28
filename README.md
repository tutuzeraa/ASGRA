# ASGRA - Attention over Scene Graphs

Framework to leverage Scene Graphs and GAT's to classify indoor scenes. Official implementation of "Attention over Scene Graphs: Indoor Scene Representations Toward CSAI Classification"


## Setup

First, you must clone the repository:

```bash
git clone git@github.com:tutuzeraa/ASGRA.git ASGRA
cd ASGRA
git submodule update --init --recursive  # to use the modified Pix2Grp
```

For installing the framework, run the following commands:

```bash
conda create -n ASGRA python=3.11 pytorch torchvision torchaudio -c pytorch -c nvidia
conda activate ASGRA 
python3 setup.py install
pip install torch_geometric
pip install -e .
```

## Datasets

We evaluate our approach in two datasets:

i) Places8
ii) RCPD

See [datasets.md](link) for more information on how to setup the datasets.

## Generating the Scene Graphs

For generating the scene graphs, we utilize this work: [Pix2Grp](https://github.com/SHTUPLUS/Pix2Grp_CVPR2024).
We did some adaptations to output the scene graphs in the format that we could process. To generate the graphs as we did, follow the instructions in [here](link).


## To run

### Training

```bash

```

### Inference

```bash

```


## Acknowledgements


## Citation

