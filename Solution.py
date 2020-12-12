import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3)
        
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        t = self.conv1(t)
        t = F.relu(t)
        
        t = self.conv2(t)
        t = F.relu(t)
        
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4(t)
        t = F.relu(t)
        
        t = self.conv5(t)
        t = F.relu(t)
        
        t = self.conv6(t)
        t = F.relu(t)
        
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t
