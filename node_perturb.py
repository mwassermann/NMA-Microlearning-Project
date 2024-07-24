""" This document:
- Creates a node perturbation network, and trains it
- then saves the weights and parameters of that network to a file
- then saves the relevant data to a .csv file """

# import dependencies
from IPython.display import Image, SVG, display
import os
from pathlib import Path
import sys
import argparse

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
      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

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

def node_perturb_test(train_images, train_labels, test_images, test_labels, indices):

  with contextlib.redirect_stdout(io.StringIO()):
    rng = np.random.default_rng(seed=12345)
    testnodeperturb = NodePerturbMLP(rng, num_hidden=15, num_inputs=784, sigma = 0.1, activation="sigmoid")

    (losses_node_perturb_test, accuracy_node_perturb_test, test_losses_node_perturb_test, snr_node_perturb_test, cosine_similarity_node_perturb_test) = testnodeperturb.train(rng, train_images, train_labels, 10, test_images[:, indices], test_labels[:, indices], learning_rate=0.001, batch_size=128, algorithm='node_perturb', noise=0.1, report=True, report_rate=1)

  # save the weights and params
  torch.save(testnodeperturb.state_dict(), "node_perturb_test.pt")

  # to load the weights and params, in the main doc:
  # netnodeperturb = MLP()
  # netnodeperturb.load_state_dict(torch.load('mlp_model.pt'))

  # save the dfs as csvs
  filenames= ["losses_node_perturb", "accuracy_node_perturb", "test_losses_node_perturb", "snr_node_perturb", "cosine_similarity_node_perturb"]
  data = [losses_node_perturb_test, accuracy_node_perturb_test, test_losses_node_perturb_test, snr_node_perturb_test, cosine_similarity_node_perturb_test]

  for i, df in enumerate(data):
    df.tofile(f"data/node_perturb/test/{filenames[i]}.csv", sep=",")

  return

def node_perturb_normal(rng, numhidden, batchsize, initweight, learnrate, noise, numepochs, activation, report, rep_rate, train_images, train_labels, test_images, test_labels, indices):
  with contextlib.redirect_stdout(io.StringIO()):
    netnodeperturb = NodePerturbMLP(rng, numhidden, num_inputs = 784, sigma=initweight, activation=activation)
    (losses_node_perturb, accuracy_node_perturb, test_losses_node_perturb, snr_node_perturb, cosine_similarity_node_perturb) = \
        netnodeperturb.train(rng, train_images, train_labels, numepochs, test_images[:, indices], test_labels[:, indices], \
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
    df.tofile(f"data/node_perturb/normal/{filenames[i]}.csv", sep=",")
  
  return

def node_perturb_online():

  # Online Learning
  with contextlib.redirect_stdout(io.StringIO()):
    netnodeperturb_online = NodePerturbMLP(rng_bp2, numhidden, num_inputs = 784, sigma=initweight, activation=activation)
    (losses_np_online, accuracy_np_online, test_loss_np_online, snr_np_online, cos_sim_np_online) = \
        netnodeperturb_online.train_online(rng_bp2, train_images, train_labels, test_images[:, indices], test_labels[:, indices], \
                      learning_rate=0.01, max_it=numupdates*batchsize, conv_loss = 1e-4, algorithm='node_perturb', noise=noise, \
                      report=report, report_rate=batchsize)
    

  # save the weights and params
  torch.save(netnodeperturb_online.state_dict(), "node_perturb_model.pt")

  # to load the weights and params, in the main doc:
  # netnodeperturb = MLP()
  # netnodeperturb.load_state_dict(torch.load('mlp_model.pt'))

  # save the dfs as csv
  filenames= ["losses_node_perturb", "accuracy_node_perturb", "test_losses_node_perturb", "snr_node_perturb", "cosine_similarity_node_perturb"]
  data = [losses_np_online, accuracy_np_online, test_loss_np_online, snr_np_online, cos_sim_np_online]

  for i, df in enumerate(data):
    df.tofile(f"data/node_perturb/online/{filenames[i]}.csv", sep=",")

  return

def node_perturb_noisy():

  # Noisy Input
  with contextlib.redirect_stdout(io.StringIO()):
    nodeperturb_noisy = NodePerturbMLP(rng_bp2, numhidden, num_inputs = 784, sigma=initweight, activation=activation)
    (losses_np_noisy, accuracy_np_noisy, test_loss_np_noisy, snr_np_noisy, cos_sim_np_noisy) = \
        nodeperturb_noisy.train(rng_bp2, train_images, train_labels, numepochs, test_images[:, indices], test_labels[:, indices], \
                        learning_rate=learnrate, batch_size=batchsize, algorithm='node_perturb', noise=noise, \
                        noise_type='gauss',report=report, report_rate=rep_rate)
    
  # save the weights and params
  torch.save(nodeperturb_noisy.state_dict(), "node_perturb_model.pt")

  # to load the weights and params, in the main doc:
  # netnodeperturb = MLP()
  # netnodeperturb.load_state_dict(torch.load('mlp_model.pt'))

  # save the dfs as csv
  filenames= ["losses_node_perturb", "accuracy_node_perturb", "test_losses_node_perturb", "snr_node_perturb", "cosine_similarity_node_perturb"]
  data = [losses_np_noisy, accuracy_np_noisy, test_loss_np_noisy, snr_np_noisy, cos_sim_np_noisy]

  for i, df in enumerate(data):
    df.tofile(f"data/node_perturb/noisy/{filenames[i]}.csv", sep=",")

  return

def node_perturb_non_stat():

  # Non-Stationary Data
  with contextlib.redirect_stdout(io.StringIO()):
    nodeperturb_nonstat = NodePerturbMLP(rng_bp2, numhidden, num_inputs = 784, sigma=initweight, activation=activation)
    (losses_np_nonstat, accuracy_np_nonstat, test_loss_np_nonstat, snr_np_non_stat, cos_sim_np_non_stat) = \
        nodeperturb_nonstat.train_nonstat_data(rng_bp2, train_images, train_labels, numepochs, test_images[:, indices], test_labels[:, indices], \
                        learning_rate=learnrate, batch_size=batchsize, algorithm='node_perturb', noise=noise, \
                        report=report, report_rate=1)
    
  # save the weights and params
  torch.save(nodeperturb_nonstat.state_dict(), "node_perturb_model.pt")

  # to load the weights and params, in the main doc:
  # netnodeperturb = MLP()
  # netnodeperturb.load_state_dict(torch.load('mlp_model.pt'))

  # save the dfs as csv
  filenames= ["losses_node_perturb", "accuracy_node_perturb", "test_losses_node_perturb", "snr_node_perturb", "cosine_similarity_node_perturb"]
  data = [losses_np_nonstat, accuracy_np_nonstat, test_loss_np_nonstat, snr_np_non_stat, cos_sim_np_non_stat]

  for i, df in enumerate(data):
    df.tofile(f"data/node_perturb/nonstat/{filenames[i]}.csv", sep=",")

  return

def main():
  # Create a CLI Parser:
  parser = argparse.ArgumentParser(description="Specify which Node Perturbation Nets to run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('nets', type=str, metavar='N', nargs="+", help = "which learning scenarios to train node perturbation networks in")

  parser.add_argument("-t", "--test", action="store_true", help ="train the network in test conditions")
  parser.add_argument("-n", "--normal", action="store_true", help="train the  network in normal conditions")
  parser.add_argument("-o", "--online", action="store_true", help="train the network in online conditions")
  parser.add_argument("-e", "--noisy", action="store_true", help="train the network in noisy conditions")
  parser.add_argument("-s", "--non-stat", action="store_true", help="train the network in non-stationary conditions")

  args = parser.parse_args()
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

  # set the random seed
  rng_np = np.random.default_rng(seed=seed)

  # select 1000 random images to test the accuracy on
  indices = rng_np.choice(range(test_images.shape[1]), size=(1000,), replace=False)

  if args["test"]:
    # Create a small newtork to test that this all actually works
    node_perturb_test(train_images=train_images, train_labels=train_labels, test_images=test_images, test_labels=test_labels, indices=indices)

  if args["normal"]:
    # create a network and train it using node perturbation in normal conditions
    node_perturb_normal(rng = rng_np, numhidden=numhidden, batchsize=batchsize, initweight=initweight, learnrate=learnrate, noise=noise, numepochs=numepochs, activation=activation, report=report, rep_rate=rep_rate, train_images=train_images, train_labels=train_labels, test_images=test_images, test_labels=test_labels, indices=indices)

  if args["online"]:
    # create a network and train it using node perturbation in online conditions
    # TODO insert code here
    a = 0

  if args["noisy"]:
    # create a network and train it using node perturbation in noisy conditions
    b = 0

  if args["non-stat"]:
    # create a network and train it using node perturbation in non-stationary conditions
    c = 0 

if __name__ == "__main__":
  main()