import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F

def train(epoch_setting, dataroot):

    workers = 4
    batch_size = 128
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = epoch_setting

    # Different learning rates for TTUR
    lr_D = 0.0001  # Discriminator learning rate (set higher)
    lr_G = 0.0004  # Generator learning rate (set lower)

    beta1 = 0.5

    ngpu = 1

    # Initial coefficient for diversity loss (used for adaptive weight adjustment)
    diversity_lambda = 0.5  # Set initial value lower
    adaptive_weight = True
    diversity_max = 10.0  # Maximum diversity weight
    diversity_step = 0.5  # Amount to increase per epoch (more gradual)


    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Encourage different features between generated images in a batch
    def diversity_loss_cosine(fake_features):
        bs = fake_features.shape[0]

        reshaped = fake_features.view(bs, -1)  # (batch_size, feature_dim)

        normed = F.normalize(reshaped, p=2, dim=1)

        cosine_sim = torch.mm(normed, normed.T)

        mask = torch.eye(bs, device=cosine_sim.device).bool()
        cosine_sim = cosine_sim.masked_fill(mask, 0)

        weighted_sim = torch.pow(cosine_sim, 2)

        return weighted_sim.sum() / (bs * (bs - 1))

    # Modified Generator class (returns intermediate feature maps)
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu

            # Initial feature extraction (latent space -> intermediate features)
            self.features_mid = nn.Sequential(
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
            )

            # Intermediate features -> final features
            self.features_final = nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
            )

            # Final output layer
            self.output = nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):
            # Extract intermediate feature maps
            mid_feat = self.features_mid(input)

            # Extract final feature maps
            final_feat = self.features_final(mid_feat)

            # Generate image
            img = self.output(final_feat)

            # Return image and two feature maps
            return img, final_feat, mid_feat


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu

            # Initial feature extraction
            self.features_init = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Intermediate feature extraction
            self.features_mid = nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Final feature extraction
            self.features_final = nn.Sequential(
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
            )

            # Output discrimination value
            self.output = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input, return_features=False):
            init_feat = self.features_init(input)
            mid_feat = self.features_mid(init_feat)
            final_feat = self.features_final(mid_feat)
            out = self.output(final_feat)

            if return_features:
                return out.view(-1), final_feat,mid_feat
            return out.view(-1)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    # Create Adam optimizers with TTUR learning rates for G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

    # Add learning rate schedulers (for performance stabilization)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=num_epochs, eta_min=lr_D * 0.1)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=num_epochs, eta_min=lr_G * 0.1)

    d_losses = []
    g_losses = []

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        # Adaptive weight adjustment (gradual increase, more fine-tuned)
        if adaptive_weight and epoch > 0:
            # Increase gradually with epochs
            diversity_lambda = min(diversity_max, diversity_lambda + diversity_step)

        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            label_real = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

            output, real_features, d_real_mid_features = netD(real_cpu, return_features=True)
            errD_real = criterion(output, label_real)

            D_x = output.mean().item()


            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake_img, fake_final_feat, fake_mid_feat = netG(noise)
            output, fake_features, d_fake_mid_features = netD(fake_img.detach(), return_features=True)
            errD_fake = criterion(output, label_fake)

            D_G_z1 = output.mean().item()

            # diversity_loss
            diversity_loss_d_real_mid = diversity_loss_cosine(d_real_mid_features)
            diversity_loss_d_fake_mid = diversity_loss_cosine(d_fake_mid_features)
            epsilon = 1e-8
            weight = diversity_loss_d_fake_mid / (diversity_loss_d_real_mid + epsilon)
            weight = torch.clamp(weight, min=1.0, max=2.0)
            penalty = diversity_lambda * weight * (diversity_loss_d_fake_mid + diversity_loss_d_real_mid)

            errD = errD_real + errD_fake + penalty
            errD.backward()
            optimizerD.step()


            netG.zero_grad()
            output, fake_features,_  = netD(fake_img, return_features=True)

            basic_g_loss = criterion(output, label_real)

            errG = basic_g_loss

            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            if i % 50 == 0:
                with torch.no_grad():
                    d_losses.append(errD.item())
                    g_losses.append(errG.item())

            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tDiv_λ: %.2f'
                    % (epoch, num_epochs, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, diversity_lambda))

        schedulerD.step()
        schedulerG.step()
