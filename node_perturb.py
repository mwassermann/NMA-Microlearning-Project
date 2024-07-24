""" This document:
- Creates a node perturbation network, and trains it
- then saves the weights and parameters of that network to a file
- then saves the relevant data to a .csv file """

# import dependencies
from IPython.display import Image, SVG, display
import os
from pathlib import Path

import random
from tqdm import tqdm
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
import contextlib
import io

## Other functions imports
from helpers import sigmoid, ReLU, add_bias, create_batches, calculate_accuracy, calculate_cosine_similarity, calculate_grad_snr

from MLP import MLP, NodePerturbMLP

# Download MNIST function
def download_mnist(train_prop=0.8, keep_prop=0.5):

  valid_prop = 1 - train_prop

  discard_prop = 1 - keep_prop

  transform = torchvision.transforms.Compose(
      [torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))]
      )


  with contextlib.redirect_stdout(io.StringIO()): #to suppress output
    
    rng_data = np.random.default_rng(seed=42)
    train_num = 50000
    shuffled_train_idx = rng_data.permutation(train_num)

    full_train_set = torchvision.datasets.MNIST(
          root="./data/", train=True, download=True, transform=transform)
    full_test_set = torchvision.datasets.MNIST(
          root="./data/", train=False, download=True, transform=transform)
    
    full_train_images = full_train_set.data.numpy().astype(float) / 255
    train_images = full_train_images[shuffled_train_idx[:train_num]].reshape((-1, 784)).T.copy()
    valid_images = full_train_images[shuffled_train_idx[train_num:]].reshape((-1, 784)).T.copy()
    test_images = (full_test_set.data.numpy().astype(float) / 255).reshape((-1, 784)).T

    full_train_labels = torch.nn.functional.one_hot(full_train_set.targets, num_classes=10).numpy()
    train_labels = full_train_labels[shuffled_train_idx[:train_num]].T.copy()
    valid_labels = full_train_labels[shuffled_train_idx[train_num:]].T.copy()
    test_labels = torch.nn.functional.one_hot(full_test_set.targets, num_classes=10).numpy().T

    train_set, valid_set, _ = torch.utils.data.random_split(
      full_train_set, [train_prop * keep_prop, valid_prop * keep_prop, discard_prop])
    test_set, _ = torch.utils.data.random_split(
      full_test_set, [keep_prop, discard_prop])

  print("Number of examples retained:")
  print(f"  {len(train_set)} (training)")
  print(f"  {len(valid_set)} (validation)")
  print(f"  {len(test_set)} (test)")

  return train_set, valid_set, test_set, train_images, valid_images, test_images, train_labels, valid_labels, test_labels

def main():
  # run download mnist function
  train_set, valid_set, test_set, train_images, valid_images, test_images, train_labels, valid_labels, test_labels = download_mnist()

  # define hyperparams
  NUM_INPUTS = 784
  NUM_OUTPUTS = 10
  numhidden = 500
  batchsize = 128
  initweight = 0.1
  learnrate = 0.001
  noise = 0.1
  numepochs = 25
  numrepeats = 1
  numbatches = int(train_images.shape[1] / batchsize)
  numupdates = numepochs * numbatches
  activation = 'sigmoid'
  report = True
  rep_rate = 1
  seed = 12345

  # Make the network
  # Train and observe the performance of NodePerturbMLP

  losses_node_perturb = np.zeros((numupdates,))
  accuracy_node_perturb = np.zeros((numepochs,))
  test_losses_node_perturb = np.zeros((numepochs,))
  snr_node_perturb = np.zeros((numepochs,))
  cosine_similarity_node_perturb = np.zeros((numepochs,))

  # set the random seed
  rng_np = np.random.default_rng(seed=seed)

  # select 1000 random images to test the accuracy on
  indices = rng_np.choice(range(test_images.shape[1]), size=(1000,), replace=False)

  # create a network and train it using weight perturbation
  with contextlib.redirect_stdout(io.StringIO()):
      netnodeperturb = NodePerturbMLP(rng_np, numhidden, num_inputs = 784, sigma=initweight, activation=activation)
      (losses_node_perturb[:], accuracy_node_perturb[:], test_losses_node_perturb[:], snr_node_perturb, cosine_similarity_node_perturb[:]) = \
          netnodeperturb.train(rng_np, train_images, train_labels, numepochs, test_images[:, indices], test_labels[:, indices], \
                              learning_rate=learnrate, batch_size=batchsize, algorithm='node_perturb', noise=noise, \
                              report=report, report_rate=rep_rate)

  # save the weights and params
  torch.save(netnodeperturb.state_dict(), "node_perturb_model.pt")

  # to load the weights and params, in the main doc:
  # netnodeperturb = MLP()
  # netnodeperturb.load_state_dict(torch.load('mlp_model.pt'))

  # save the dfs as csv
  filenames= ["losses_node_perturb", "accuracy_node_perturb", "test_losses_node_perturb", "snr_node_perturb", "cosine_similarity_node_perturb"]
  data = [losses_node_perturb, accuracy_node_perturb, test_losses_node_perturb, snr_node_perturb, cosine_similarity_node_perturb]

  for i, df in enumerate(data):
    df.tofile(f"data/node_perturb/{filenames[i]}.csv", sep=",")

if __name__ == "__main__":
  main()