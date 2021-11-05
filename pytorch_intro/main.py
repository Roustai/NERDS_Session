import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## train data
class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

#read in files
file_in = os.getcwd() + "/heart_data/heart.csv"
df = pd.read_csv(file_in)
X = df.iloc[:, 0:-1] #grab first 13 elements for input
y = df.iloc[:, -1]   #seperate last element (target values)


#reshape data from a standard dataframe to a numpy array
scaler = StandardScaler()
X_train = scaler.fit_transform(X)

#reshape data into a tensor of floats
train_data = trainData(torch.FloatTensor(X_train),torch.FloatTensor(y))
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

#build model and pass it to device (CPU or GPU)
model = NeuralNetwork().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def accuracy(y_pred, y_test):
    #compare predicting values vs actual values
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

epochs = 5
for e in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    print(f"Epoch {e + 1}\n-------------------------------")
    for features, labels in train_loader:

        features, labels = features.to(device), labels.to(device)
        y_pred = model(features)
        loss = loss_fn(y_pred, labels.unsqueeze(1))
        acc = accuracy(y_pred, labels.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

print("Done Training!")
