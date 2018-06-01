import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-MMD')
parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=8, help='hidden dimension of z (default: 8)')
parser.add_argument('-LAMBDA', type=float, default=1, help='regularization coef MMD term (default: 1)')
parser.add_argument('-n_channel', type=int, default=1, help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
args = parser.parse_args()

trainset = MNIST(root='./data/',
                 train=True,
                 transform=transforms.ToTensor(),
                 download=True)

testset = MNIST(root='./data/',
                 train=False,
                 transform=transforms.ToTensor(),
                 download=True)

train_loader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=testset,
                         batch_size=104,
                         shuffle=False)

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(1024, self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU()
        )
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1, bias=False),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(112 ** 2, 28 ** 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x).view(-1, 112 ** 2)
        x = self.fc(x).view(-1, 1, 28, 28)
        return x

def rbf_kernel(X, Y):
    batch_size = X.size()[0]

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = torch.exp(-exponent / (2 * args.sigma))
    return K[:batch_size, :batch_size], K[:batch_size, batch_size:], K[batch_size:, batch_size:]

def lorentz_kernel(X, Y):
    batch_size = X.size()[0]

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    C = 2 * args.n_z * (args.sigma ** 2)
    K = C / (C + exponent ** 2)

    return K[:batch_size, :batch_size], K[:batch_size, batch_size:], K[batch_size:, batch_size:]

encoder, decoder = Encoder(args), Decoder(args)
criterion = nn.MSELoss()

encoder.train()
decoder.train()

if torch.cuda.is_available():
    encoder, decoder = encoder.cuda(), decoder.cuda()

if torch.cuda.device_count() > 1:
    encoder = nn.DataParallel(encoder, device_ids=range(torch.cuda.device_count()))
    decoder = nn.DataParallel(decoder, device_ids=range(torch.cuda.device_count()))

one = torch.Tensor([1])
mone = one * -1

if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr = args.lr)
dec_optim = optim.Adam(decoder.parameters(), lr = args.lr)

enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)

for epoch in range(args.epochs):
    step = 0
    for (images, _) in tqdm(train_loader):

        if torch.cuda.is_available():
            images = images.cuda()

        enc_optim.zero_grad()
        dec_optim.zero_grad()

        # ======== Train Generator ======== #

        images = Variable(images)
        batch_size = images.size()[0]

        z = encoder(images)
        x_recon = decoder(z)

        recon_loss = criterion(x_recon, images)
        recon_loss.backward(one)

        # ======== MMD Kernel Loss ======== #

        z_fake = Variable(torch.randn(images.size()[0], args.n_z) * args.sigma)
        if torch.cuda.is_available():
            z_fake = z_fake.cuda()

        z_real = encoder(Variable(images.data))

        mmd_real, mmd_cross, mmd_fake = lorentz_kernel(z_real, z_fake)
        (mmd_fake + mmd_real + mmd_cross).mean().backward(one)

        enc_optim.step()
        dec_optim.step()

        step += 1

        if (step + 1) % 300 == 0:
            print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
                  (epoch + 1, args.epochs, step + 1, len(train_loader), recon_loss.data.item()))

    if (epoch + 1) % 1 == 0:
        batch_size = 104
        test_iter = iter(test_loader)
        test_data = next(test_iter)

        z_real = encoder(Variable(test_data[0]).cuda())
        reconst = decoder(torch.randn_like(z_real)).cpu().view(-1, 1, 28, 28)

        if not os.path.isdir('./data/reconst_images'):
            os.makedirs('data/reconst_images')

        save_image(test_data[0].view(-1, 1, 28, 28), './data/reconst_images/wae_mmd_input.png')
        save_image(reconst.data, './data/reconst_images/wae_mmd_images_%d.png' % (epoch + 1))