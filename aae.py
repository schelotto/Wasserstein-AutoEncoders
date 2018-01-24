import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch MNIST AAE')

parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')

args = parser.parse_args()

X_dim = 28 ** 2
h_dim = 1000
z_dim = 2
EPSILON = 1e-15

trainset = MNIST(root='./data/',
                 train=True,
                 transform=transforms.ToTensor(),
                 download=False)

testset = MNIST(root='./data/',
                 train=False,
                 transform=transforms.ToTensor(),
                 download=False)

train_loader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=testset,
                         batch_size=args.batch_size,
                         shuffle=False)

class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(X_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3gauss = nn.Linear(h_dim, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        xgauss = self.lin3gauss(x)
        return xgauss

class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, X_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return F.sigmoid(x)

class D_net(nn.Module):
    def __init__(self):
        super(D_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))

Q, P, D = Q_net(), P_net(), D_net()
Q.train()
P.train()
D.train()

# Optimizers
P_optim = optim.Adam(P.parameters(), lr = 0.001)
Q_enc_optim = optim.Adam(Q.parameters(), lr = 0.001)
Q_gen_optim = optim.Adam(Q.parameters(), lr = 0.001)
D_optim = optim.Adam(D.parameters(), lr = 0.001)

for epoch in range(args.epochs):
    step = 0
    for i, (images, _) in enumerate(train_loader):
        P.zero_grad()
        Q.zero_grad()
        D.zero_grad()

        images = Variable(images)
        batch_size = images.size()[0]
        images = images.view(batch_size, -1)

        z_sample = Q(images)
        x_sample = P(z_sample)
        recon_loss = F.binary_cross_entropy(x_sample + EPSILON, images + EPSILON)
        recon_loss.backward()

        P_optim.step()
        Q_enc_optim.step()

        Q.eval()
        z_real_gauss = Variable(torch.randn(images.size()[0], z_dim) * 5.)
        D_real_gauss = D(z_real_gauss)

        z_fake_gauss = Q(images)
        D_fake_gauss = D(z_fake_gauss)

        D_loss = -torch.mean(torch.log(D_real_gauss + EPSILON) + torch.log(1 - D_fake_gauss + EPSILON))
        D_loss.backward()
        D_optim.step()

        Q.train()
        z_fake_gauss = Q(images)
        D_fake_gauss = D(z_fake_gauss)

        G_loss = -torch.mean(torch.log(D_fake_gauss + EPSILON))
        G_loss.backward()
        Q_gen_optim.step()

        step += 1

        if (step + 1) % 100 == 0:
            print("Epoch: %d, Step: [%d/%d], Reconstruction Loss: %.4f, Discriminator Loss: %.4f, Generator Loss: %.4f" %
                  (epoch + 1, step + 1, len(train_loader), recon_loss.data[0], D_loss.data[0], G_loss.data[0]))

    P.eval()
    z1 = np.arange(-10, 10, 2.).astype('float32')
    z2 = np.arange(-10, 10, 2.).astype('float32')
    nx, ny = len(z1), len(z2)
    recons_image = []

    for z1_ in z1:
        for z2_ in z2:
            x = P(Variable(torch.from_numpy(np.asarray([z1_, z2_]))).view(-1, z_dim)).view(1, 1, 28, 28)
            recons_image.append(x)

    if not os.path.isdir('./data/reconst_images'):
        os.makedirs('data/reconst_images')
    save_image(recons_image.data, './data/reconst_images/aae_images_%d.png' % (epoch+1), nrow=nx)
