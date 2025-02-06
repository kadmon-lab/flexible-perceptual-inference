# Flexible Perceptual Inference

This project combines theoretical modeling, neural network training, and rodent behavioral experiments to study inference-driven adaptive decision-making in dynamic and uncertain environments.

## Installation

To install, first clone the repo to your local machine and ```cd``` into the project directory. 

Create a new virtual enviroment using Python's venv (tested with Python 3.10)
```
python -m venv .venv
source .venv/bin/activate
```

(If venv is not available and you don't have root rights, install miniconda, activate its base environment, and use its Python to create a new *Python* venv (not a conda environment)). You can check the python that is being used by running ```python --version```. 


Now, install the heavy computing frameworks. Pytorch and Jax are best installed by following the instructions on your platform that are available on the web. It might be that GPU support for Pytorch and Jax is difficult to maintain. In that case, only install GPU support for one of them, and use the CPU for the other. Further, try to use the bundled CUDA version of Pytorch, as it is easier to maintain.

```
pip install -e .
```

## Train networks
To train 10 networks of each variant run:
```
python enzyme/src/main/train_trained_LSTM.py 
python enzyme/src/main/train_random_LSTM.py 
python enzyme/src/main/train_random_CTX_LSTM.py
```

## Collect testing data
To collect and save testing data from all agents run:
```
python enzyme/src/mouse_task/get_data.py -checkpointing = True
```