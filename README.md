# ASGRA - Attention over Scene Graphs 🔎🕸️

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

- **Places8**
- **RCPD**

See [datasets.md](https://github.com/tutuzeraa/ASGRA/blob/main/asgra/datasets/datasets.md) for more information on how to setup the datasets.

## Generating the Scene Graphs

For generating the scene graphs, we utilize this work: [Pix2Grp](https://github.com/SHTUPLUS/Pix2Grp_CVPR2024).
We did some adaptations to output the scene graphs in the format that we could process. To generate the graphs as we did, follow the instructions in [here](https://github.com/tutuzeraa/Pix2Grp_CVPR2024/tree/a8e9fbb4c4c798c0dd456d1570ff1a524c004a50?tab=readme-ov-file#instructions).


## To run

To train and evaluate the model, you can run the following commands: 

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python3 asgra/main.py -m train -c configs/asgra_best.json -w 8 -o results/run1
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python3 asgra/main.py -m eval -c configs/asgra_best.json -w 8 -o results/eval-run1 --weights path-to-trained-weights
```

## Acknowledgements

This repository is built over [Pix2Grp](https://github.com/SHTUPLUS/Pix2Grp_CVPR2024), that is built over [LAVIS](https://github.com/salesforce/LAVIS) and [SGTR](https://github.com/Scarecrow0/sgtr). We would like to thank them for their great open-source code and models.


## Citation

