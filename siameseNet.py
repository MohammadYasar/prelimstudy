import os, sys, glob
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from sklearn import preprocessing


num_epochs = 6
num_classes = 10
batch_size = 100
learning_rate = 0.001
segment = 6

class VectorizeData(Dataset):
    def __init__(self, df_path):
        self.df = np.asarray(pd.read_csv(df_path, error_bad_lines=False))
        self.vec = self.df[:,1:self.df.shape[1]-1]
        min_max_scaler = preprocessing.MinMaxScaler()
        self.vec = min_max_scaler.fit_transform(self.vec)
        #self.vec = np.reshape(self.vec, (-1,2))
        self.out = self.df[:,self.df.shape[1]-1]
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X = self.vec[idx]
        y = self.out[idx]

        return X.reshape(1,-1),y

# create instance of custom dataset
DATA_PATH = os.path.abspath(os.path.dirname(sys.argv[0])) + "/transcriptions/error{}.csv".format(segment)
valDATA_PATH = os.path.abspath(os.path.dirname(sys.argv[0])) + "/transcriptions/error{}val.csv".format(segment)
ds = VectorizeData(DATA_PATH)
dl = DataLoader(dataset=ds, batch_size=3)
vs = VectorizeData(valDATA_PATH)
vdl = DataLoader(dataset=vs, batch_size=1)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,64, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2= nn.Sequential(
            nn.Conv1d(64,32,kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(288, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet()
#print (model.parameters)
loss_fn = nn.CrossEntropyLoss()
N=1000
train_accuracies = []
train_losses = []
val_accuracies = []
val_losses = []

optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.9)
for epoch in range(0, N):
      correct = 0.0
      cum_loss = 0.0
      model.train()
      batchSize = 5
      dl = DataLoader(ds, batch_size=batchSize, shuffle=True)
      it = iter(dl)
      print ("len of batch {}".format(len(ds)))
      for i in range(len(it)):
          xs,ys =  next(it)
          inputs = xs#.cuda()
          labels = torch.tensor(ys, dtype=torch.long)#.cuda()
          scores = model(inputs.float())
          loss = loss_fn(scores, labels)
          # Count how many correct in this batch.
          max_scores, max_labels = scores.max(1)
          correct += (max_labels == labels).sum().item()
          cum_loss += loss.item()

          # Zero the gradients in the network.
          optimizer.zero_grad()

          #Backward pass. (Gradient computation stage)
          loss.backward()

          # Parameter updates (SGD step) -- if done with torch.optim!
          optimizer.step()

          # Parameter updates (SGD step) -- if done manually!
          # for param in model.parameters():
          #   param.data.add_(-learningRate, param.grad)

          # Logging the current results on training.
          if (i + 1) % 100 == 0:
              print('Train-epoch %d. Iteration %05d, Avg-Loss: %.4f, Accuracy: %.4f' %
                    (epoch, i + 1, cum_loss/(i + 1), correct / ((i + 1) * batchSize)))

          train_accuracies.append(correct / len(it))
          train_losses.append(cum_loss / (i + 1))

      # Make a pass over the validation data.
      correct = 0.0
      cum_loss = 0.0
      model.eval()
      vit = iter(vdl)
      for i in range(len(vit)):
          xvs, yvs = next(vit)
          inputs = xvs
          labels = torch.tensor(yvs, dtype=torch.long)
          # Forward pass. (Prediction stage)
          scores = model(inputs.float())
          cum_loss += loss_fn(scores, labels).item()

           # Count how many correct in this batch.
          max_scores, max_labels = scores.max(1)
          correct += (max_labels == labels).sum().item()

      val_accuracies.append(correct / len(vit))
      val_losses.append(cum_loss / (i + 1))

      # Logging the current results on validation.
      print('Validation-epoch %d. Avg-Loss: %.4f, Accuracy: %.4f' %
            (epoch, cum_loss / (i + 1), correct / len(vit)))
