from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import numpy as np
import csv
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data

import NeuralSim.dcgan as dcgan
import NeuralSim.utils as myutils


# DATASET
class NPYDataset(torch.utils.data.Dataset):

    def __init__(self, filename, n_facies=3):
        self.data = np.load(filename)
        self.n_facies = n_facies
        print("\nDataset loaded")
        print("Shape:", self.data.shape)
        print("Unique values:", np.unique(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        facies = self.data[idx]
        # (64,64) -> (64,64,3)
        onehot = np.eye(self.n_facies, dtype=np.float32)[facies]
        # (64,64,3) -> (3,64,64)
        onehot = np.transpose(onehot, (2, 0, 1))
        return torch.from_numpy(onehot)

# MAIN
if __name__ == "__main__":

    FileNotFoundError = IOError
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='npy_slices')
    parser.add_argument('--dataroot', default='training_dataset/training_windows_T1.npy')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batchSize', type=int, default=256) # 256 suggested by copilot
    parser.add_argument('--imageSize', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--nz', type=int, default=60)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int,default=64)
    parser.add_argument('--niter', type=int, default=200) # 200 suggested by copilot
    parser.add_argument('--lrD', type=float,default=5e-5)
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
    parser.add_argument('--experiment', default='geo_facies_gan')
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0')
    opt = parser.parse_args()

    print(opt)

    os.makedirs(opt.experiment,exist_ok=True)

    # ==========================================================
    # RANDOM SEED
    # ==========================================================
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

    # ==========================================================
    # DATASET
    # ==========================================================

    dataset = NPYDataset(filename=opt.dataroot, n_facies=opt.nc)

    sample = dataset[0]
    print("\nSanity checks")
    print("Sample shape:", sample.shape)
    print("Sample dtype:", sample.dtype)
    print("Min:", sample.min())
    print("Max:", sample.max())

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True
    )

    batch = next(iter(dataloader))
    print("\nBatch shape:", batch.shape)

    # ==========================================================
    # GENERATOR
    # ==========================================================

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

    netG.apply(
        myutils.weights_init
    )

    if opt.netG:

        netG.load_state_dict(
            torch.load(opt.netG)
        )

        print("Loaded Generator")

    # ==========================================================
    # DISCRIMINATOR
    # ==========================================================

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

    netD.apply(
        myutils.weights_init
    )

    if opt.netD:

        netD.load_state_dict(
            torch.load(opt.netD)
        )

        print("Loaded Discriminator")

    # ==========================================================
    # LATENT VECTORS
    # ==========================================================

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

    # ==========================================================
    # OPTIMIZERS
    # ==========================================================

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


    # ==========================================================
    # TRAINING LOOP
    # ==========================================================

    gen_iterations = 0

    lossD_history = []
    lossG_history = []

    loss_csv = os.path.join(opt.experiment, "loss_history.csv")

    with open(loss_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "iteration",
            "lossD",
            "lossG"
        ])


    for epoch in range(opt.niter):

        data_iter = iter(dataloader)

        i = 0

        while i < len(dataloader):

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

                # FIXED
                x_real = data.to(device)

                netD.zero_grad()

                errD_real = netD(
                    x_real
                ).mean()

                x_fake = netG(
                    noise_vec.normal_(0, 1)
                ).detach()

                errD_fake = netD(
                    x_fake
                ).mean()

                errD = -(
                    errD_real -
                    errD_fake
                )

                errD.backward()

                optimizerD.step()

            for p in netD.parameters():
                p.requires_grad = False

            netG.zero_grad()

            x_fake = netG(
                noise_vec.normal_(0, 1)
            )

            errG = -netD(
                x_fake
            ).mean()

            errG.backward()

            optimizerG.step()

            gen_iterations += 1

            # Store losses at every generator update
            lossD_history.append(errD.item())
            lossG_history.append(errG.item())

            with open(loss_csv, "a", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    epoch,
                    gen_iterations,
                    errD.item(),
                    errG.item()
                ])

            print(
                f'[{epoch}/{opt.niter-1}]'
                f'[{i}/{len(dataloader)}]'
                f'[{gen_iterations}] '
                f'Loss_D: {errD.item():.5f} '
                f'Loss_G: {errG.item():.5f}'
            )

        if epoch % 5 == 0:

            # --------------------------------------------------
            # Save networks
            # --------------------------------------------------

            torch.save(
                netG.state_dict(),
                f'{opt.experiment}/netG_epoch_{epoch}.pth'
            )

            torch.save(
                netD.state_dict(),
                f'{opt.experiment}/netD_epoch_{epoch}.pth'
            )

            # --------------------------------------------------
            # Create epoch folder
            # --------------------------------------------------

            epoch_dir = os.path.join(
                opt.experiment,
                f"epoch_{epoch:03d}"
            )

            os.makedirs(
                epoch_dir,
                exist_ok=True
            )

            # --------------------------------------------------
            # Generate 10 realizations
            # --------------------------------------------------

            with torch.no_grad():

                noise = torch.randn(
                    10,
                    opt.nz,
                    1,
                    1,
                    device=device
                )

                fake = netG(noise)

                # Collapse one-hot channels

                facies = torch.argmax(
                    fake,
                    dim=1
                )

            # --------------------------------------------------
            # Save raw facies arrays
            # --------------------------------------------------

            np.save(
                os.path.join(
                    epoch_dir,
                    "generated_facies.npy"
                ),
                facies.cpu().numpy()
            )

            # --------------------------------------------------
            # Geological color map
            # --------------------------------------------------

            colors = np.array([
                [0.4, 0.4, 0.4],   # facies 0
                [1.0, 0.9, 0.1],   # facies 1
                [0.0, 0.6, 1.0]    # facies 2
            ])

            # --------------------------------------------------
            # Save 10 samples
            # --------------------------------------------------

            for k in range(10):

                img = facies[k].cpu().numpy()

                rgb = colors[img]

                plt.figure(figsize=(4,4))

                plt.imshow(
                    rgb,
                    interpolation="nearest"
                )

                plt.axis("off")

                plt.tight_layout()

                plt.savefig(
                    os.path.join(
                        epoch_dir,
                        f"sample_{k:02d}.png"
                    ),
                    bbox_inches="tight",
                    pad_inches=0
                )

                plt.close()

            # --------------------------------------------------
            # Save losses
            # --------------------------------------------------

            np.save(
                os.path.join(
                    epoch_dir,
                    "lossD.npy"
                ),
                np.array(lossD_history)
            )

            np.save(
                os.path.join(
                    epoch_dir,
                    "lossG.npy"
                ),
                np.array(lossG_history)
            )

            # --------------------------------------------------
            # Save loss plot
            # --------------------------------------------------

            plt.figure(figsize=(8,4))

            plt.plot(
                lossD_history,
                label="Loss D"
            )

            plt.plot(
                lossG_history,
                label="Loss G"
            )

            plt.legend()

            plt.xlabel(
                "Generator iteration"
            )

            plt.ylabel(
                "Loss"
            )

            plt.grid(True)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    epoch_dir,
                    "losses.png"
                )
            )

            plt.close()

            print(
                f"Checkpoint saved at epoch {epoch}"
            )