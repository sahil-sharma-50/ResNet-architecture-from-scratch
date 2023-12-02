import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
csv_path = 'data.csv'
tab = pd.read_csv(csv_path, sep=';')
rand = np.random.randint(1, 100)
train_data, val_data = train_test_split(tab, test_size=0.20, random_state=rand)


# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train_loader = DataLoader(ChallengeDataset(train_data, 'train'), batch_size=64, shuffle=True)
val_loader = DataLoader(ChallengeDataset(val_data, 'val'), batch_size=64)


# create an instance of our ResNet model
# TODO
model = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
criterion = t.nn.MSELoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)
trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, cuda=True, early_stopping_patience=25)


# go, go, go... call fit on trainer
res = trainer.fit(epochs=50)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
