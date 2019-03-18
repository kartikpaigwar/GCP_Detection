import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import cv2

train_on_gpu = torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1568*4, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 1568*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x


# create a complete CNN
model = Net()
model.load_state_dict(torch.load('model3_99%.pt'))
model.eval()

if train_on_gpu:
    model.cuda()

img = cv2.imread("l2.png",0)
img_arr = np.ascontiguousarray(img, dtype=np.float32) / 255
img_arr = (img_arr-0.5)/0.5
data = torch.from_numpy(img_arr)
data = data.unsqueeze(0)
data = data.unsqueeze(0).cuda()
output = model(data)
probout = output.cpu()
prob = probout.detach().numpy()
prob = np.exp(prob)
print(prob)
_, pred = torch.max(output, 1)
print(pred.data.cpu().numpy())
