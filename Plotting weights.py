from __future__ import print_function
import matplotlib.pyplot as plt

import torch
import numpy as np
from VAE3 import Encoder, Decoder, VAE

density = {}


def get_weights(model, density_level):
    transpose_enc_mu = torch.transpose(model.enc.mu.weight, 0, 1)
    transpose_enc_var = torch.transpose(model.enc.var.weight, 0, 1)
    encoder_weights = torch.cat((model.enc.linear.weight, transpose_enc_mu, transpose_enc_var), 1)
    encoder_weights = encoder_weights.detach().numpy()
    encoder_weights_no_zeros = encoder_weights[np.nonzero(encoder_weights)]
    density[f'encoder_weights_{density_level}'] = encoder_weights_no_zeros


INPUT_DIM = 28 * 28  # size of each input
HIDDEN_DIM = 512  # hidden dimension
LATENT_DIM = 2  # latent vector dimension

encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

# Initialize Decoder
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

# Set directory
directory = '/Users/petervanderwal/Capstone_code/Sparse_VAE/Final/Combining Graphs'

# Initialize VAE
model50 = VAE(encoder, decoder)

data50 = torch.load('/Users/petervanderwal/Capstone_code/Sparse_VAE/Final/model_50%/5 '
                    'Epochs/Models/model.pt')
model50.load_state_dict(data50['state_dict'])



get_weights(model50, 50)


model95 = VAE(encoder, decoder)
data95 = torch.load('/Users/petervanderwal/Capstone_code/Sparse_VAE/Final/model_95%/5 '
                    'Epochs/Models/model.pt')
model95.load_state_dict(data95['state_dict'])
get_weights(model95, 95)


model5 = VAE(encoder,decoder)
data5 = torch.load('/Users/petervanderwal/Capstone_code/Sparse_VAE/Final/model_5%/5 '
                   'Epochs/Models/model.pt')
model5.load_state_dict(data5['state_dict'])
get_weights(model5, 5)


plt.figure(figsize=[10, 8])
plt.hist(density['encoder_weights_5'], bins=150, label='5% Sparsity', alpha=1, color='red', range=(-0.5, 0.5), density = True)
plt.hist(density['encoder_weights_50'], bins=150, label='50% Sparsity', alpha=0.8, color='blue', range=(-0.5, 0.5), density = True)
plt.hist(density['encoder_weights_95'], bins=150, label='95% Sparsity', alpha=0.6, color='green', range=(-0.5, 0.5), density = True)
plt.legend(loc='best')
plt.xlabel('Weight', fontsize=10)
plt.ylabel('Frequency of the weight', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title('Encoder weight histogram', fontsize=10)
plt.savefig(directory + '/Weight Histogram')
plt.show(block=False)
plt.pause(20)
plt.close()
