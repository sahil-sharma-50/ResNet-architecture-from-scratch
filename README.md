# Classification using PyTorch

This project implements and trains a variation of the widely used architecture, ResNet, for classifying images from solar panels. The goal is to identify defects in solar panels, which are subject to degradation due to various factors such as transport, wind, hail, etc. The defects can be of different types, such as cracks or inactive regions.

## Dataset

Solar modules are composed of cells and the images in the dataset represent these cells.

The dataset is provided in the `data.csv` file, which contains the following columns:
- `filename`: The name of the image file.
- `crack`: A binary indicator (0 or 1) specifying whether the solar cell has a crack.
- `inactive`: A binary indicator (0 or 1) specifying whether the solar cell is inactive.

## Files in the Repository

- `data.csv`: The dataset file.
- `data.py`: A Python script that transforms the data.
- `environment.yml`: A file that lists the dependencies required to run the project.
- `images.zip`: A zip file containing all the images in the dataset.
- `model.py`: A Python script that contains the ResNet50 model.
- `train.py`: A Python script to train the model.
- `trainer.py`: A Python script that contains the training loop.

## How to Run the Project

1. Clone the repository.
2. Install the dependencies listed in the `environment.yml` file.
3. Unzip the `images.zip` file to get the images.
4. Run the `data.py` script to transform the data.
5. Run the `train.py` script to start training the model.
