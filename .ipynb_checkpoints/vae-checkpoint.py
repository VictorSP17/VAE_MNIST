# Prerequisites:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import sys
from torch.distributions.multivariate_normal import MultivariateNormal
import os
import matplotlib.pyplot as plt
import pylab
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Hyper-parameters:
bs = 100
num_epochs = 0
NN = True

# Image preprocessing:
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform = transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform = transform, download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

##############################################################################################
# Model, networks, losses, training (defined as functions!):
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    # Reparametrization trick :)
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var) # 0.5 perque es std, no var!
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

vae = VAE(x_dim=784, h_dim1= 512, h_dim2=256, z_dim=2)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters())

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data #.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    trainL[epoch-1] = train_loss / len(train_loader.dataset)
    
    if (epoch+1) % 5 == 0:
        torch.save(vae.state_dict(), os.path.join('save', 'vae{}.ckpt'.format(epoch+1)))

def test(epoch):
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data #.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    
    test_loss /= len(test_loader.dataset)
    testL[epoch-1] = test_loss
    print('====> Test set loss: {:.4f}'.format(test_loss))

""
# Load
state_dict = torch.load('save/vae6.ckpt')
vae.load_state_dict(state_dict)

testL = np.zeros(num_epochs)
trainL = np.zeros(num_epochs)

# TRAINING!
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test(epoch)
    
    # VISUALS
    plt.figure()
    pylab.xlim(0, num_epochs + 1)
    pylab.ylim(100, 170)
    plt.plot(range(1, num_epochs + 1), trainL, label='train loss')
    plt.plot(range(1, num_epochs + 1), testL, label='test loss')
    plt.legend()
    plt.savefig('save/loss.pdf')
    plt.close()

with torch.no_grad():
    z = torch.randn(64, 2) #.cuda()
    sample = vae.decoder(z).round() #.cuda()
    save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')


# NEAREST NEIGHBOURS:
# Images 28x28, search the closest one.
# function(generated_image) --> closest training_image
if NN == True:
    aux_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                  batch_size=1,
                                                  shuffle=False)
    def nearest_gt(generated_image):
        min_d = 0
        closest = False
        for i, (image, _) in enumerate(aux_data_loader):
            image = image.view(1, 28, 28).round() # all distances in binary...
            d = torch.dist(generated_image,image) # must be torch tensors (1,28,28)
            if i == 0 or d < min_d:
                min_d = d
                closest = image

        return closest

    # calculate closest to...
    z = torch.randn(24, 2) #.cuda()
    sample = vae.decoder(z).round() #.cuda()
    sample = sample.view(24, 1, 28, 28)
    save_image(sample, './samples/f24.png')
    NN = torch.zeros(24, 1, 28, 28)
    for i in range(0,24):
        NN[i] = nearest_gt(sample[i])
        print(i)
    save_image(NN.data, './samples/NN24.png')





