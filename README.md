<div align="center">

# lightning-template

</div>

Template for machine learning projects with [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), focussed on allowing research projects to get set up quickly. The template includes an MNIST example, logging with weights and biases, and a recommended style guide. This README serves as a style guide and gives some general advice for getting the most out of this template.

**To start a new project, create a new repo with the "Use this template" button. Upon doing do, the initial-commit.yml action will run - deleting this README and replacing it with ```project-README.md```**

## Template features

This template is designed to give the user the minimum amount of useful features to get started with their research project quickly. Naturally, there is a tradeoff between adding features and introducing bloat and I have strived to make the template as minimal as possible, whilst also providing some example code and some boilerplate code. Here are the main features this template comes with:

1. MNIST + ResNet example. By default, the template will include an MNIST DataModule and a ResNet LightningModule (and running train.py will use these out-of-the-box). These are common enough that they should be useful, and even if not, they provide useful examples of how to properly make models and datasets in Pytorch Lightning. Feel free to remove them if not needed.

2. Logging with Weights and Biases. By default, ```train.py``` initialises a Weights and Biases logger, and uses two of the included logging callbacks. Note: I intend to create a package with lots of useful logging callbacks at some point in the future, so these may be moved.

3. Black format check GitHub action. On pushes and pull requests to main an action will run to check if the code is black formatted.

4. Auto-generated README. When creating a repo with this template, this README will be deleted and replaced with ```project-specific-README.md```. The action also performs a find and replace to insert the actual name of the repository where "REPO-NAME" is currently. The Action then self-destructs to ensure it never runs again and does not use up your actions minutes. If you have any ideas for anything else that would be useful to trigger when the repo is created, let me know :).

Note on CI testing: this template used to contain some standard tests for checking that your models are working properly. These have since been removed, and in the future, I may create a separate package for ML model and dataset testing. If you want some inspiration for setting up CI testing for you model, the most recent commit which still has these tests is ```f8887bf512f37ad099781e30b29d3c726cbfa967```.


## Using the CLI

We use the Python argument parser to create a CLI for training models. It is recommended to use this as much as possible, as it makes it easier to run hyperparameter sweeps and reduces the need to keep editing code between runs. The MNIST + ResNet examples demonstrate how to use argparse args with datasets and models. This is done with the add_argparse_args classmethods. Note: it is technically possible to remove the named arguments from the class initialsers and simply pass ```vars(args)``` into ```__init__(**kwargs)```, however, this template opts to keep the named arguments and pass ```args.named_argument``` to make it more explicit when initialising the classes.

The pl.Trainer module also has a powerful CLI and lets you control things like GPUs, floating point precision, logging and more from the command line.

## Using the MNIST+ResNet example

It is simple to use the default example model out-of-the-box; for training, show available options by running:

```bash
python src/train.py --help
```

This will give a list of command line arguments you can use in the program - e.g. ```python src/train.py --max_epochs 50 --batch_size 8 --log_every_n_steps 5 --learning_rate 0.01```.

This project integrates with [Weights and Biases](https://wandb.ai/site) for logging and it is strongly recommended to use it (it's free!). By default, including the ```--logger True``` flag in the CLI will use Weights and Biases.
When using Weights and Biases on a new machine, run ```wandb login``` in the terminal, and paste the API key from your weights and biases account when prompted to set it up.
