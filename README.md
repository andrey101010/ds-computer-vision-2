# ds-computer-vision-2
Welcome to the GitHub repository for exploring the exciting field of computer vision using Tensorflow on the MNIST dataset. My project aims to demonstrate the capabilities of these powerful deep learning frameworks by training and evaluating various models on this widely-used dataset. Additionally, I have integrated MLflow to monitor and track the performance of the models, providing an efficient pipeline for fine-tuning and optimizing the deep learning parameters. I hope you find this repository informative and useful in your own computer vision projects.

If you want to see PyTorch in action use this link: [Pytorch](https://github.com/andrey101010/ds-computer-vision)

Computer-Vision-2

## Environment 
Use the [requirements](requirements.txt) file in this repo to create a new environment. 
We have to install hdf5:

```BASH
 brew install hdf5
 brew install graphviz
```
With the system setup like that, we can go and create our environment and install tensorflow

```BASH
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
```
If you already have hdf5
```BASH
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2
```
otherwise, if you have just installed hdf5 with brew, then
```BASH
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.2_2
```

```BASH
pip install -U pip
pip install --no-binary=h5py h5py
pip install -r requirements.txt
```
