Checklist for starting project:

* [ ] If necessary, merge the CI testing branch to main to get some initial pytest features
* [ ] Create initial Milestones e.g. EDA, create dataset, create model, train initial model, etc.
* [ ] Add custom issue labels (for example severity of issue, (more) issue types, and staus of issue (abandoned, in progress, etc.)
* [ ] Update README, including CI testing badge


<div align="center">

# TITLE

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Blog](http://img.shields.io/badge/Blog-NameofPost-c044ce.svg)](https://charl-ai.github.io/)
[![Kaggle](http://img.shields.io/badge/Kaggle-CompetitionName-44c5ce.svg)](https://www.kaggle.com/competitions)


![CI testing](https://github.com/Charl-AI/lightning-template/workflows/CI%20testing/badge.svg?branch=main&event=push)


</div>

## Description
Template for machine learning projects. The template includes an MNIST example, logging with weights and biases, and a reusable test suite. To use the test suite, include all branches of the template when making a new project and merge the CI testing branch into main.

This project uses [PytorchLightning](https://pytorch-lightning.readthedocs.io/en/latest/) to organise the codebase and provide some useful abstractions.


## Installation
Note: this project has been developed and tested on Ubuntu. The project may run on other platforms, however, this is not guaranteed. If you do not have an Ubuntu machine, consider using WSL.

First, clone the repo
```bash
# clone project
git clone https://github.com/Charl-AI/REPO-NAME

# change to project directory
cd REPO-NAME
```

A virtual environment is recommended for this project. Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt --no-cache-dir
```
## Data
The data used in the default example is the standard MNIST dataset. It will download automatically to ```data/``` by default.


## Running

For training, show available options by running:

```bash
python src/train.py --help
```

This will give a list of command line arguments you can use in the program - e.g. ```python src/train.py --max_epochs 50 --batch_size 1 --log_every_n_steps 5 --learning_rate 0.01```.

This project integrates with [Weights and Biases](https://wandb.ai/site) for logging and it is strongly recommended to use it (it's free!). By default, including the ```--logger True``` flag in the CLI will use Weights and Biases.
When using Weights and Biases on a new machine, run ```wandb login``` in the terminal, and paste the API key from your weights and biases account to set it up.

### Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
