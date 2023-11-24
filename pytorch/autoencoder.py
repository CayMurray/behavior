## MODULES ##

from torch import utils,nn,optim
from torchvision import datasets
from torchvision import transforms


## PREPARE DATA ##

transforms = transforms.ToTensor()

data = datasets.MNIST(root = './data/mnist',
                      train = True,
                      download = False,
                      transform = transforms)

loader = utils.data.DataLoader(dataset=data,
                                batch_size=32,
                                shuffle=True)


## DEFINE MODEL ##

class AE(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,36),
            nn.ReLU(),
            nn.Linear(36,18),
            nn.ReLU(),
            nn.Linear(18,9)
            )

        self.decoder = nn.Sequential(
            nn.Linear(9,18),
            nn.ReLU(),
            nn.Linear(18,36),
            nn.ReLU(),
            nn.Linear(36,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28)
            )
        
    def forward(self,x):
        x = x.view(x.size[0],-1)
        encoded = self.encoder(x)
        decoded = self.decoded(encoded)
        return decoded 


## TRAIN MODEL ##

model = AE()
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),
                       lr = 1e-1,
                       weight_decay=1e-8)

epochs = 20

for epoch in range(epochs):
    for (batch,_) in loader:
        pass
    