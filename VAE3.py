from __future__ import print_function
import os
import shutil
import time
import argparse
import matplotlib.pyplot as plt
import logging
import hashlib
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

import core
from core import Masking, CosineDecay
from utils import get_mnist_dataloaders


INPUT_DIM = 28 * 28  # size of each input
HIDDEN_DIM = 512  # hidden dimension
LATENT_DIM = 2  # latent vector dimension
logger = None


class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''

    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''

    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]

        return predicted


class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''

    def __init__(self, enc, dec, save_features=False, bench_model=False):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        m.bias.data.fill_(0.01)


def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density_level,
                                               hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train(args, model, train_loader, optimizer, epoch, lr_scheduler, mask=None):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for batch_idx, (x, _) in enumerate(train_loader):

        # reshape the data into [batch_size, 784]
        x = x.view(-1, 28 * 28)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_sample, z_mu, z_var = model(x)

        # reconstruction loss
        recon_loss = F.binary_cross_entropy(x_sample, x, reduction='sum')

        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

        # total loss
        loss = recon_loss + kl_loss

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader) * args.batch_size,
                       100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader) * args.batch_size

    return train_loss


def evaluate(args, model, data_loader):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0
    correct = 0
    n = 0
    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, _) in enumerate(data_loader):
            # reshape the data
            x = x.view(-1, 28 * 28)

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = F.binary_cross_entropy(x_sample, x, reduction='sum')

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

            # total loss
            loss = recon_loss + kl_loss
            test_loss += loss.item()

    test_loss /= len(data_loader) * args.batch_size

    return test_loss


def plot_loss(args, epochs, train, test, val, num_param, directory):
    plt.figure()
    plt.plot(epochs[:], train[:], label='Training loss', color='blue')
    plt.plot(epochs[:], test[:], label='Test loss', color='red')
    plt.plot(epochs[:], val[:], label='Validation loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Binary Cross Entropy')
    plt.title(f'lr = {args.lr}, SD = 0.01, # param. = {num_param}')

    plt.legend(loc='upper right')

    plt.savefig(directory + '/Loss function')


def create_directories(args):
    main_dir = f'/Users/petervanderwal/Capstone_code/Sparse_VAE/Final/model_{round((1 - args.density_level) * 100)}%'
    if args.epochs == 1:
        epoch_dir = main_dir + f'/{args.epochs} Epoch'
    else:
        epoch_dir = main_dir + f'/{args.epochs} Epochs'
    directory_models = epoch_dir + '/Models'
    directory_graphs = epoch_dir + '/Graphs'
    directory_generated_images = epoch_dir + '/Images'

    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    if not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir)
    if not os.path.exists(directory_models):
        os.makedirs(directory_models)
    if not os.path.exists(directory_graphs):
        os.makedirs(directory_graphs)
    if not os.path.exists(directory_generated_images):
        os.makedirs(directory_generated_images)
    return directory_models, directory_graphs, directory_generated_images


def save_model(args, state, directory):
    model_dir = directory + '/model.pt'
    torch.save(state, model_dir)


def number_non_zero_param(parameters):
    non_zero_param = 0
    for p in parameters:
        non_zero_param += p.nonzero().size(0)
    return non_zero_param


def generate_images(model, number_of_images, directory):
    for i in range(number_of_images):
        z = torch.randn(1, LATENT_DIM)
        reconstructed_img = model.dec(z)
        img = reconstructed_img.view(28, 28).data
        i_directory = directory + f'/image_{i}.png'
        save_image(img, i_directory)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--save-features', action='store_true',
                        help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--save-model', type=str, default='./models/model.pt', help='For Saving the current Model')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--model', type=str, default='vae')
    parser.add_argument('--density_level', type=float, default=0.1)
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--bench', action='store_true',
                        help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--decay-schedule', type=str, default='cosine',
                        help='The decay schedule for the pruning rate. Default: cosine. Choose from: cosine, linear.')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--iters', type=int, default=2,
                        help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--sparse', default=False)
    core.add_sparse_args(parser)
    args = parser.parse_args()

    setup_logger(args)
    print_and_log(args)

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        # Load data
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)

        # Initialize Encoder
        encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

        # Initialize Decoder
        decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

        # Initialize VAE
        model = VAE(encoder, decoder)

        # Initialize weight (RandomNormal)
        init_weights(model)

        # Set optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, args.decay_frequency, gamma=0.1)

        mask = None
        if args.sparse:
            decay = CosineDecay(args.prune_rate, len(train_loader) * args.epochs)
            mask = Masking(optimizer, decay)
            mask.add_module(model, density=args.density_level)

        epoch_count = []
        train_loss_per_epoch = []
        test_loss_per_epoch = []
        val_loss_per_epoch = []
        best_test_loss = float('inf')

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            epoch_count.append(epoch)
            best_test_loss = float('inf')

            # Training function
            train_loss = train(args, model, train_loader, optimizer, epoch, lr_scheduler, mask)

            test_loss = evaluate(args, model, test_loader)

            train_loss_per_epoch.append(train_loss)
            test_loss_per_epoch.append(test_loss)

            if args.valid_split > 0.0:
                val_acc = evaluate(args, model, valid_loader)
                val_loss_per_epoch.append(val_acc)

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()},
                            is_best=False, filename=args.save_model)

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(
                optimizer.param_groups[0]['lr'], time.time() - t0))

            print(f'Epoch {epoch}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

            if args.sparse and epoch < args.epochs+1:
                mask.at_end_of_epoch()

        # Creates the directories to save the model, the graphs, and the generated images
        m_dir, g_dir, i_dir = create_directories(args)

        # Returns the number of non zero parameters after training
        non_zero_param = number_non_zero_param(model.parameters())

        # Saves model and relevant variables
        save_model(args, {'epochs': args.epochs,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'learning rate': args.lr,
                          'train loss per epoch': train_loss_per_epoch,
                          'test loss per epoch': test_loss_per_epoch,
                          'validation loss per epoch': val_loss_per_epoch,
                          '# of parameters': non_zero_param,
                          'Density': args.density_level}, m_dir)

        # Plots the loss function of the train-, test-, and validation set versus the number of epochs.
        plot_loss(args, epoch_count[:], train_loss_per_epoch[:], test_loss_per_epoch[:],
                  val_loss_per_epoch[:], non_zero_param, g_dir)

        # Generates images by randomly sampling from the Latent Space
        generate_images(model, 10, i_dir)

        print(non_zero_param)

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        args.density_level += 0.1
        args.sparse = True
        print(f'New density level: {args.density_level}')
        print_and_log("\nIteration end: {0}/{1}\n".format(i + 1, args.iters))


if __name__ == '__main__':
    main()
