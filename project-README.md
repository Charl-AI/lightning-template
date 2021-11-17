<div align="center">

# REPO-NAME

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.0000.0000-B31B1B.svg)](https://www.nature.com/articles/nature14539) -->
<!-- [![Conference](http://img.shields.io/badge/NeurIPS-20XX-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018) -->
<!-- [![Blog](http://img.shields.io/badge/Blog-NameofPost-c044ce.svg)](https://charl-ai.github.io/) -->
<!-- [![Kaggle](http://img.shields.io/badge/Kaggle-CompetitionName-44c5ce.svg)](https://www.kaggle.com/competitions) -->

</div>

## Executive Summary
Summary of research project and repository purpose (it can be nice to write this before you start coding to make your ideas and purpose clearer).


This project uses [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) to organise the codebase and provide some useful abstractions.

## Installation
Note: this project has been developed and tested on Ubuntu. The project may run on other platforms, however, this is not guaranteed. If you do not have an Ubuntu machine, consider using WSL.

It is recommended to clone the repo with the GitHub CLI:
```bash
# clone project
gh repo clone Charl-AI/REPO-NAME

# change to project directory
cd REPO-NAME
```

A virtual environment is recommended for this project. Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt --no-cache-dir
```

If you are having CUDA issues, you can find the Pytorch builds for each version of CUDA [here](https://pytorch.org/get-started/locally/). For example, an NVIDIA RTX 3090 uses the CUDA 11.3 compute platform, this can be installed by running:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Data
The data used in the default example is the standard MNIST dataset. It will download automatically to ```data/``` by default.


## Training

For training, show available options by running:

```bash
python src/train.py --help
```

This will give a list of command line arguments you can use in the program - e.g. ```python src/train.py --max_epochs 50 --batch_size 8 --log_every_n_steps 5 --learning_rate 0.01```.

This project integrates with [Weights and Biases](https://wandb.ai/site) for logging and it is strongly recommended to use it (it's free!). By default, including the ```--logger True``` flag in the CLI will use Weights and Biases.
When using Weights and Biases on a new machine, run ```wandb login``` in the terminal, and paste the API key from your weights and biases account when prompted to set it up.

## Inference and visualisation

We use notebooks for inference and visualisation to improve readability. These can be found in ```src/experiments/```.

<!-- ## Citation -->
<!-- ``` -->
<!-- @article{YourName, -->
  <!-- title={Your Title}, -->
  <!-- author={Your team}, -->
  <!-- journal={Location}, -->
  <!-- year={Year} -->
<!-- } -->
<!-- ``` -->
