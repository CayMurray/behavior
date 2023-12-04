## MODULES AND DEVICES ##

import torch
import torch.nn.functional as F
from torch import utils,nn,optim
from torchvision import datasets
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## PREPARE DATA ##

transforms = transforms.Compose([transforms.ToTensor()])

data = datasets.MNIST(root = './data/mnist',
                      train = True,
                      download = False,
                      transform = transforms)

loader = utils.data.DataLoader(dataset=data,
                                batch_size=32,
                                shuffle=True)


## DEFINE MODEL ##

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 28, 28) 
        return x
    

## TRAIN MODEL ##

model = AutoEncoder()
model.to(device)

loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)
epochs = 20

for epoch in range(epochs):
    epoch_loss = []

    for (image,_) in loader:
        image = image.to(device)
        reconstruction = model.forward(image)
        criterion = loss(reconstruction,image)
        optimizer.zero_grad()
        criterion.backward(),optimizer.step()
        epoch_loss.append(criterion.item())

    average_loss = torch.tensor(epoch_loss,dtype=torch.float).mean()
    print(f'Epoch {epoch} loss: {average_loss}')

torch.save(model,'models/mnist_autoencoder.pth')
#model = torch.load('models/mnist_autoencoder.pth')


        