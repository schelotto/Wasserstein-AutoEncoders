import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')
parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=8, help='hidden dimension of z (default: 8)')
parser.add_argument('-LAMBDA', type=float, default=1, help='regularization coef GAN term (default: 1)')
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
            nn.Conv2d(1, self.dim_h, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.dim_h),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.dim_h, 2 * self.dim_h, 5, stride=2, padding=2),
            nn.BatchNorm2d(2 * self.dim_h),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * self.dim_h, 4 * self.dim_h, 5, stride=2, padding=2),
            nn.BatchNorm2d(4 * self.dim_h),
            nn.LeakyReLU(0.2, True)
        )
        self.fc = nn.Linear(self.dim_h * 4 ** 3, self.n_z)

    def forward(self, x):

        input = x.view(-1, 1, 28, 28)
        x = self.main(input)
        x = x.view(-1, self.dim_h * 4 ** 3)
        output = self.fc(x)

        return output

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.flatten = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 4 ** 3),
            nn.LeakyReLU(0.2, True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.dim_h, 2 * self.dim_h, 5),
            nn.BatchNorm2d(2 * self.dim_h),
            nn.LeakyReLU(0.2, True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.dim_h, self.dim_h, 5),
            nn.BatchNorm2d(self.dim_h),
            nn.LeakyReLU(0.2, True),
        )
        self.deconv_out = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h, 1, 8, stride=2),
            nn.Sigmoid()
        )

    def forward(self,
                input: torch.Tensor):
        output = self.flatten(input)
        output = output.view(-1, 4 * self.dim_h, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)

        return output.view(-1, 28 ** 2)

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.n_z, self.dim_h * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(self.dim_h, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.n_z, 1, 1)
        x = self.main(x)
        return x

encoder, decoder, discriminator = Encoder(args), Decoder(args), Discriminator(args)
criterion = nn.MSELoss()

encoder.train()
decoder.train()
discriminator.train()

# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr = args.lr)
dec_optim = optim.Adam(decoder.parameters(), lr = args.lr)
dis_optim = optim.Adam(discriminator.parameters(), lr = 0.5 * args.lr)

enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
dis_scheduler = StepLR(dis_optim, step_size=30, gamma=0.5)

if torch.cuda.is_available():
    encoder, decoder, discriminator = encoder.cuda(), decoder.cuda(), discriminator.cuda()

one = torch.Tensor([1])
mone = one * -1

if torch.cuda.is_available():
    one = one.cuda()
    mone = mone.cuda()

for epoch in range(args.epochs):
    step = 0

    for images, _ in tqdm(train_loader):

        if torch.cuda.is_available():
            images = images.cuda()

        images = images.view(-1, 28 ** 2)
        encoder.zero_grad()
        decoder.zero_grad()
        discriminator.zero_grad()

        # ======== Train Discriminator ======== #

        frozen_params(decoder)
        frozen_params(encoder)
        free_params(discriminator)

        z_fake = Variable(torch.randn(images.size()[0], args.n_z) * args.sigma)
        if torch.cuda.is_available():
            z_fake = z_fake.cuda()
        d_fake = discriminator(z_fake)

        z_real = encoder(Variable(images.data))
        d_real = discriminator(z_real)

        torch.log(d_fake).mean().backward(mone)
        torch.log(1 - d_real).mean().backward(mone)

        dis_optim.step()

        # ======== Train Generator ======== #

        free_params(decoder)
        free_params(encoder)
        frozen_params(discriminator)

        batch_size = images.size()[0]

        z_real = encoder(images)
        x_recon = decoder(z_real)
        d_real = discriminator(encoder(Variable(images.data)))

        recon_loss = criterion(x_recon, images)
        d_loss = args.LAMBDA * (torch.log(d_real)).mean()

        recon_loss.backward(one)
        d_loss.backward(mone)

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

        z_real_ = encoder(Variable(test_data[0]).cuda())
        reconst = decoder(z_real_).cpu().view(batch_size, 1, 28, 28)

        if not os.path.isdir('./data/reconst_images'):
            os.makedirs('data/reconst_images')

        save_image(test_data[0].view(batch_size, 1, 28, 28), './data/reconst_images/wae_gan_input.png')
        save_image(reconst.data, './data/reconst_images/wae_gan_images_%d.png' % (epoch + 1))