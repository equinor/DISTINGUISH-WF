from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data

import NeuralSim.dcgan as dcgan
import NeuralSim.utils as myutils

class NPY(torch.utils.data.Dataset):

    def __init__(self, filename):
        self.data = np.load(filename)
        print("Dataset loaded")
        print("Shape:", self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        facies = self.data[idx]
        onehot = np.eye(3, dtype=np.float32)[facies]
        onehot = np.transpose(onehot,(2, 0, 1))
        return torch.from_numpy(onehot)

if __name__ == "__main__":

    FileNotFoundError = IOError

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='grdecl')
    parser.add_argument('--dataroot', default='models')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--imageSize', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--nz', type=int, default=60)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=4000)
    parser.add_argument('--lrD', type=float, default=5e-5)
    parser.add_argument('--lrG', type=float, default=5e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--netG', default='')
    parser.add_argument('--netD', default='')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5)
    parser.add_argument('--noBN', action='store_true', default=False)
    parser.add_argument('--n_extra_layers', type=int, default=0)
    parser.add_argument('--experiment', default='geo_grdecl')
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--maxSamples', type=int, default=1000000)
    parser.add_argument('--maxFiles', type=int, default=50)
    parser.add_argument('--strideX', type=int, default=32)
    parser.add_argument('--strideY', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')

    opt = parser.parse_args()

    print(opt)

    opt.experiment += f"_{opt.strideX}_{opt.strideY}"
    opt.experiment += f"_{opt.maxFiles}_{opt.nz}"

    os.makedirs(opt.experiment, exist_ok=True)

    # ---------------------------------------------------------
    # Random seed
    # ---------------------------------------------------------
    opt.manualSeed = 1234

    print("Random Seed:", opt.manualSeed)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    try:
        print("===========================")
        print("GPUs:", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            print("device", i, torch.cuda.get_device_name(i))

    except Exception:
        pass

    device = torch.device(
        opt.device if torch.cuda.is_available()
        else "cpu"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    if opt.dataset == "grdecl":

        dataset = myutils.CustomDatasetFromGRD(
            filename=opt.dataroot,
            height=opt.imageSize,
            width=opt.imageSize,
            channels=opt.nc,
            constant_axis=0,
            do_flip=False,
            stride_x=opt.strideX,
            stride_y=opt.strideY,
            max_samples=opt.maxSamples,
            max_files=opt.maxFiles
        )

    elif opt.dataset == "npy_slices":
        dataset = NPY(
            filename=opt.dataroot,
            n_facies=opt.nc
        )

    else:
        raise Exception("Unknown dataset")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )

    # ---------------------------------------------------------
    # Networks
    # ---------------------------------------------------------

    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(
            opt.imageSize,
            opt.nz,
            opt.nc,
            opt.ngf,
            opt.ngpu,
            opt.n_extra_layers
        ).to(device)
    else:
        netG = dcgan.DCGAN_G(
            opt.imageSize,
            opt.nz,
            opt.nc,
            opt.ngf,
            opt.ngpu,
            opt.n_extra_layers
        ).to(device)

    netG.apply(myutils.weights_init)

    if opt.netG:
        netG.load_state_dict(torch.load(opt.netG))
        print("Loaded Generator")

    print(netG)

    if opt.noBN:
        netD = dcgan.DCGAN_D_nobn(
            opt.imageSize,
            opt.nz,
            opt.nc,
            opt.ndf,
            opt.ngpu,
            opt.n_extra_layers
        ).to(device)

    else:
        netD = dcgan.DCGAN_D(
            opt.imageSize,
            opt.nz,
            opt.nc,
            opt.ndf,
            opt.ngpu,
            opt.n_extra_layers
        ).to(device)

    netD.apply(myutils.weights_init)

    if opt.netD:
        netD.load_state_dict(torch.load(opt.netD))
        print("Loaded Discriminator")

    print(netD)

    # ---------------------------------------------------------
    # Latent vectors
    # ---------------------------------------------------------

    noise_vec = torch.FloatTensor(
        opt.batchSize,
        opt.nz,
        1,
        1
    ).to(device)

    fixed_noise_vec = torch.FloatTensor(
        opt.batchSize,
        opt.nz,
        1,
        1
    ).normal_(0, 1).to(device)

    fixed_zero_vec = torch.FloatTensor(
        opt.batchSize,
        opt.nz,
        1,
        1
    ).normal_(0, 1e-2).to(device)

    # ---------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------

    if opt.adam:

        optimizerD = optim.Adam(
            netD.parameters(),
            lr=opt.lrD,
            betas=(opt.beta1, 0.999)
        )

        optimizerG = optim.Adam(
            netG.parameters(),
            lr=opt.lrG,
            betas=(opt.beta1, 0.999)
        )

    else:

        optimizerD = optim.RMSprop(
            netD.parameters(),
            lr=opt.lrD
        )

        optimizerG = optim.RMSprop(
            netG.parameters(),
            lr=opt.lrG
        )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------

    gen_iterations = 0

    for epoch in range(opt.niter):

        data_iter = iter(dataloader)

        i = 0

        while i < len(dataloader):

            # -----------------------------------------
            # Train Critic
            # -----------------------------------------

            for p in netD.parameters():
                p.requires_grad = True

            if (
                (opt.netD == '' and gen_iterations < 25)
                or
                (gen_iterations % 500 == 0)
            ):
                Diters = 100
            else:
                Diters = opt.Diters

            j = 0

            while j < Diters and i < len(dataloader):

                j += 1

                for p in netD.parameters():
                    p.data.clamp_(
                        opt.clamp_lower,
                        opt.clamp_upper
                    )

                try:
                    data = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    data = next(data_iter)

                i += 1

                x_real = data[0].to(device)

                netD.zero_grad()

                errD_real = netD(x_real).mean()

                x_fake = netG(
                    noise_vec.normal_(0, 1)
                ).detach()

                errD_fake = netD(x_fake).mean()

                errD = -(errD_real - errD_fake)

                errD.backward()

                optimizerD.step()

            # -----------------------------------------
            # Train Generator
            # -----------------------------------------

            for p in netD.parameters():
                p.requires_grad = False

            netG.zero_grad()

            x_fake = netG(noise_vec.normal_(0, 1))

            errG = -netD(x_fake).mean()

            errG.backward()

            optimizerG.step()

            gen_iterations += 1

            print(
                f'[{epoch}/{opt.niter-1}]'
                f'[{i}/{len(dataloader)}]'
                f'[{gen_iterations}] '
                f'Loss_D: {errD.item():.5f} '
                f'Loss_G: {errG.item():.5f} '
                f'D_real: {errD_real.item():.5f} '
                f'D_fake: {errD_fake.item():.5f}'
            )

            if gen_iterations % 100 == 0:

                myutils.save_image(
                    x_real,
                    f'{opt.experiment}/real_samples.png',
                    device
                )

                myutils.save_image(
                    netG(fixed_noise_vec),
                    f'{opt.experiment}/fake_samples.png',
                    device
                )

                myutils.save_image(
                    netG(fixed_zero_vec),
                    f'{opt.experiment}/fake_zero.png',
                    device
                )

        if epoch % 111 == 0:

            torch.save(
                netG.state_dict(),
                f'{opt.experiment}/netG_epoch_{epoch}.pth'
            )

            torch.save(
                netD.state_dict(),
                f'{opt.experiment}/netD_epoch_{epoch}.pth'
            )

            print(f"Checkpoint saved at epoch {epoch}")