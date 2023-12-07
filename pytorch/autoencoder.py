## MODULES AND DEVICES ##

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd 
from torch import utils,nn,optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## CLASSES AND FUNCTIONS ##

class DfToTensor():

    def __call__(self,data):
        self.data = data 
        tensor = torch.tensor(self.data.to_numpy(),dtype=torch.float32)

        return tensor


class NormalizeTensor():

    def __call__(self,data):
        self.data = data
        self.min = data.min()
        self.max = data.max()

        return (self.data-self.min)/(self.max-self.min)

    
class LoadedData(Dataset):

    def __init__(self,data,transforms):
        super().__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        sample = self.data.iloc[idx,:]
        transformed_tensor = self.transforms(sample)

        return transformed_tensor


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(115,89),
            nn.ReLU(),
            nn.Linear(89,64),
            nn.ReLU(),
            nn.Linear(64,12),
            nn.ReLU(),
            nn.Linear(12,2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2,12),
            nn.ReLU(),
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,89),
            nn.ReLU(),
            nn.Linear(89,115),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        #x = x.view(x.size(0), 1, 28, 28) 
        return x
    
def visualize_performance(epoch_losses):
    fig,ax = plt.subplots(figsize=(20,10))
    ax.plot(range(len(epoch_losses)),epoch_losses)
    ax.set_xticks(range(0,epochs,5))
    ax.set_xlabel('Epochs',fontsize=20,labelpad=20)
    ax.set_ylabel('Loss',fontsize=20,labelpad=20)
    plt.show()
    

## LOAD DATA ##

mnist_transforms = transforms.Compose([transforms.ToTensor()])

data = datasets.MNIST(root = './data/mnist',
                      train = True,
                      download = False,
                      transform = mnist_transforms)

loader = utils.data.DataLoader(dataset=data,
                                batch_size=32,
                                shuffle=True)


transformations = transforms.Compose([DfToTensor(),NormalizeTensor()])
rat_data = LoadedData(pd.read_csv('data/FS_raw.csv').T,transformations)
train_loader = DataLoader(rat_data,batch_size=32,shuffle=True)


## DEFINE MODEL AND PARAMS

model = AutoEncoder()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)
epochs = 20
overall_loss = []


## TRAINING LOOP ##

for epoch in range(epochs):
    epoch_loss = []

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        reconstructed = model.forward(batch)
        loss = criterion(reconstructed,batch)
        loss.backward(),optimizer.step()
        epoch_loss.append(loss.item())

    overall_loss.append(torch.tensor(epoch_loss,dtype=torch.float32).mean())
    print(f'Epoch {epoch} loss: {overall_loss[-1]}')
    

## VISUALIZE ##

visualize_performance(overall_loss)


## GET LATENT SPACE COMPONENTS ##

with torch.no_grad():
    df = pd.read_csv('data/FS_raw.csv')
    context_ids = df.T.index.tolist()
    input_data = torch.tensor(df.T.to_numpy(),dtype=torch.float32).to(device)
    encoded_data = model.encoder(input_data).cpu()
    reduced_components = np.array(encoded_data,dtype=np.float32)
    df_reduced = pd.DataFrame(data=reduced_components,index=context_ids,columns=['AE1','AE2'])
    df_reduced['context_ids'] = context_ids
    df_reduced.to_csv('data/FS_ae.csv',index=False)




