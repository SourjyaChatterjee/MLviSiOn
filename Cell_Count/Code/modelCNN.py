import torch.nn as nn
import torch.nn.functional as F
 
# define the NN architecture
class Conv_count(nn.Module):
    def __init__(self):
        super(Conv_count, self).__init__()
 
        #(256,256,3)-->(256,256,6) convolution
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        
        #(128,128,6)-->(128,128,16) convolution
        self.conv2 = nn.Conv2d(6,16,3, padding=1)
        
        #(64,64,16)--> (64,64,32) convolution
        self.conv3 = nn.Conv2d(16,32,3, padding=1)
        
        # 32*32*32 --> 1200 dense layer
        self.fc1 = nn.Linear(32*32*32, 1200)
 
        # 1200 --> 120 dense layer
        self.fc2 = nn.Linear(1200, 120)
 
        # 120 --> 84 dense layer
        self.fc3 = nn.Linear(120, 84)
 
        # 84 --> 10 dense layer
        self.fc4 = nn.Linear(84, 10)
 
        # 10 --> 1 dense layer
        self.fc5 = nn.Linear(10, 1)
 
        # max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        
        # dropout layer (p=0.20)
        self.dropout = nn.Dropout(0.20)
 
        self.double()
 
 
    def forward(self, x):
 
      # add sequence of convolutional and max pooling layers  
      block1 = self.pool(F.relu(self.conv1(x)))
      block2 = self.pool(F.relu(self.conv2(block1)))
      block3 = self.pool(F.relu(self.conv3(block2)))
 
      # flatten image input
      y = block3.view(-1, 32*32*32)
      # add dropout layer
      y = self.dropout(y)
      # add 1st hidden layer, with relu activation function
      dense1 = F.relu(self.fc1(y))
      # add dropout layer
      dense1 = self.dropout(dense1)
      # add 2nd hidden layer, with relu activation function
      dense2 = F.relu(self.fc2(dense1))
      # add dropout layer
      dense2 = self.dropout(dense2)
      # add 3rd hidden layer, with relu activation function
      dense3 = F.relu(self.fc3(dense2))
      # add dropout layer
      dense3 = self.dropout(dense3)
      # add 4th hidden layer, with relu activation function
      dense4 = F.relu(self.fc4(dense3))
      # add dropout layer
      dense4 = self.dropout(dense4)
      # Output layer
      dense5 = F.relu(self.fc5(dense4))
                
      return dense5
 
