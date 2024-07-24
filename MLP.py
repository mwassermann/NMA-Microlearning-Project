# dependencies
from IPython.display import Image, SVG, display
import os
from pathlib import Path

import random
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
import contextlib
import io

from helpers import sigmoid, ReLU, add_bias, create_batches, calculate_accuracy, calculate_cosine_similarity, calculate_grad_snr

# The main network class
# This will function as the parent class for our networks, which will implement different learning algorithms
class MLP(object):
    """
    The class for creating and training a two-layer perceptron.
    """

    # The initialization function
    def __init__(self,
                rng,
                num_hidden,
                num_inputs = 784,
                num_outputs = 10,
                sigma=1.0, activation='sigmoid'):
        """
        The initialization function for the MLP.

         - N is the number of hidden units
         - sigma is the SD for initializing the weights
         - activation is the function to use for unit activity, options are 'sigmoid' and 'ReLU'
        """

        # store the variables for easy access

        super().__init__()

        self.N = num_hidden
        self.sigma = sigma
        self.activation = activation

        # initialize the weights
        #self.W_h = rng.normal(scale=self.sigma, size=(self.num_hidden, num_inputs + 1))  # input-to-hidden weights & bias
        #self.W_y = rng.normal(scale=self.sigma, size=(num_outputs, self.num_hidden + 1))  # hidden-to-output weights & bias
        #self.B = rng.normal(scale=self.sigma, size=(self.num_hidden, num_outputs))  # feedback weights

        self.W_h_1 = rng.normal(scale=self.sigma, size=(self.N, num_inputs+1))  # input-to-hidden weights & bias
        self.W_h_2 = rng.normal(scale=self.sigma, size=(self.N , self.N + 1))  # hidden-to-hidden weights & bias

        self.W_y = rng.normal(scale=self.sigma, size=(num_outputs, self.N + 1))  # hidden-to-output weights & bias
        self.B = rng.normal(scale=self.sigma, size=(self.N, num_outputs))  

    def _store_initial_weights_biases(self):
        """
        Stores a copy of the network's initial weights and biases.

        Note: NOT CURRENTLY USED ANYWHERE
        """

        self.init_lin1_weight = self.lin1.weight.data.clone()
        self.init_lin2_weight = self.lin2.weight.data.clone()
        if self.bias:
            self.init_lin1_bias = self.lin1.bias.data.clone()
            self.init_lin2_bias = self.lin2.bias.data.clone()
        

    def _set_activation(self):
        """
        Sets the activation function used for the hidden layer.
        """

        if self.activation_type.lower() == "sigmoid":
            self.activation = torch.nn.Sigmoid() # maps to [0, 1]
        elif self.activation_type.lower() == "tanh":
            self.activation = torch.nn.Tanh() # maps to [-1, 1]
        elif self.activation_type.lower() == "relu":
            self.activation = torch.nn.ReLU() # maps to positive
        elif self.activation_type.lower() == "identity":
            self.activation = torch.nn.Identity() # maps to same
        else:
            raise NotImplementedError(
                f"{self.activation_type} activation type not recognized. Only "
                "'sigmoid', 'relu' and 'identity' have been implemented so far.")

    # The non-linear activation function
    def activate(self, inputs):
        """
        Pass some inputs through the activation function.
        """
        if self.activation == 'sigmoid':
            Y = sigmoid(inputs)
        elif self.activation == 'ReLU':
            Y = ReLU(inputs)
        else:
            raise Exception("Unknown activation function")
        return Y

    # The function for performing a forward pass up through the network during inference
    def inference(self, rng, inputs, W_h_1=None, W_h_2=None, W_y=None, noise=0.):
        """
        Recognize inputs, i.e. do a forward pass up through the network. If desired, alternative weights
        can be provided
        """

        # load the current network weights if no weights given
        if W_h_1 is None:
            W_h_1 = self.W_h_1
        if W_h_2 is None:
            W_h_2 = self.W_h_2
        if W_y is None:
            W_y = self.W_y

        # calculate the hidden activities
        hidden1 = self.activate(np.dot(W_h_1, add_bias(inputs)))
        if not (noise == 0.):
            hidden1 += rng.normal(scale=noise, size=hidden1.shape)

        hidden2 = self.activate(np.dot(W_h_2, add_bias(hidden1)))
        if not (noise == 0.):
            hidden2 += rng.normal(scale=noise, size=hidden2.shape)

        # calculate the output activities
        output = self.activate(np.dot(W_y, add_bias(hidden2)))

        if not (noise == 0.):
            output += rng.normal(scale=noise, size=output.shape)

        return hidden1, hidden2, output

    def forward(self, X, y=None):
        """
        Runs a forward pass through the network.

        Arguments:
        - X (torch.Tensor): Batch of input images.
        - y (torch.Tensor, optional): Batch of targets. This variable is not used
        here. However, it may be needed for other learning rules, to it is
        included as an argument here for compatibility.

        Returns:
        - y_pred (torch.Tensor): Predicted targets.

        Note: NOT CURRENTLY USED ANYWHERE
        """

        h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
        y_pred = self.softmax(self.lin2(h))
        return y_pred

    # A function for calculating the derivative of the activation function
    def act_deriv(self, activity):
        """
        Calculate the derivative of some activations with respect to the inputs
        """
        if self.activation == 'sigmoid':
            derivative = activity * (1 - activity)
        elif self.activation == 'ReLU':
            derivative = 1.0 * (activity > 1)
        else:
            raise Exception("Unknown activation function")
        return derivative

    def mse_loss_batch(self, rng, inputs, targets, W_h=None, W_y=None, output=None):
        """
        Calculate the mean-squared error loss on the given targets (average over the batch)
        """

        # do a forward sweep through the network
        if (output is None):
            (hidden1, hidden2, output) = self.inference(rng, inputs)
        return np.sum((targets - output) ** 2, axis=0)

    # The function for calculating the mean-squared error loss
    def mse_loss(self, rng, inputs, targets, W_h=None, W_y=None, output=None):
        """
        Calculate the mean-squared error loss on the given targets (average over the batch)
        """
        return np.mean(self.mse_loss_batch(rng, inputs, targets, W_h=W_h, W_y=W_y, output=output))

    # function for calculating perturbation updates
    def perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for perturbation learning, using noise with SD as given
        """
        raise NotImplementedError()

    def node_perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for node perturbation learning, using noise with SD as given
        """
        raise NotImplementedError()

    # function for calculating gradient updates
    def gradient(self, rng, inputs, targets):
        """
        Calculates the weight updates for gradient descent learning
        """

        # do a forward pass
        hidden1, hidden2, output = self.inference(rng, inputs)

        # calculate the gradients
        error = targets - output

    # calculate delta for the output layer
        delta_y = error * self.act_deriv(output)
    
    # calculate delta for the second hidden layer
        delta_h2 = np.dot(self.W_y[:, :-1].T, delta_y) * self.act_deriv(hidden2)
        
        # calculate delta for the first hidden layer
        delta_h1 = np.dot(self.W_h_2[:, :-1].T, delta_h2) * self.act_deriv(hidden1)
        
        # calculate gradients
        delta_W_y = np.dot(delta_y, add_bias(hidden2).T)
        delta_W_h_2 = np.dot(delta_h2, add_bias(hidden1).T)
        delta_W_h_1 = np.dot(delta_h1, add_bias(inputs).T)

        return delta_W_h_1, delta_W_h_2, delta_W_y

    # function for calculating feedback alignment updates
    def feedback(self, rng, inputs, targets):
        """
        Calculates the weight updates for feedback alignment learning
        """
        raise NotImplementedError()

    # function for calculating Kolen-Pollack updates
    def kolepoll(self, rng, inputs, targets, eta_back=0.01):
        """
        Calculates the weight updates for Kolen-Polack learning
        """
        raise NotImplementedError()

    def return_grad(self, rng, inputs, targets, algorithm='backprop', eta=0.01, noise=1.0):
        # calculate the updates for the weights with the appropriate algorithm
        if algorithm == 'perturb':
            delta_W_h_1, delta_W_h_2, delta_W_y = self.perturb(rng, inputs, targets, noise=noise)
        elif algorithm == 'node_perturb':
            delta_W_h_1, delta_W_h_2, delta_W_y = self.node_perturb(rng, inputs, targets, noise=noise)
        elif algorithm == 'feedback':
            delta_W_h_1, delta_W_h_2, delta_W_y = self.feedback(rng, inputs, targets)
        elif algorithm == 'kolepoll':
            delta_W_h_1, delta_W_h_2, delta_W_y = self.kolepoll(rng, inputs, targets, eta_back=eta)
        else:
            delta_W_h_1, delta_W_h_2, delta_W_y = self.gradient(rng, inputs, targets)

        return delta_W_h_1, delta_W_h_2, delta_W_y

    # function for updating the network
    def update(self, rng, inputs, targets, algorithm='backprop', eta=0.01, noise=1.0):
        """
        Updates the synaptic weights (and unit biases) using the given algorithm, options are:

        - 'backprop': backpropagation-of-error (default)
        - 'perturb' : weight perturbation (use noise with SD as given)
        - 'feedback': feedback alignment
        - 'kolepoll': Kolen-Pollack
        """

        delta_W_h_1, delta_W_h_2, delta_W_y = self.return_grad(rng, inputs, targets, algorithm=algorithm, eta=eta, noise=noise)

        # do the updates
        self.W_h_1 += eta * delta_W_h_1
        self.W_h_2 += eta * delta_W_h_2
        self.W_y += eta * delta_W_y
    
    def alter_inputs(self, train,type=None):
        if type == "gauss":
            row = len(train)
            mean = 0
            var = 0.1
            sigma = var**0.5
            for i in range(train.shape[1]):
                gauss = np.random.normal(mean,sigma,(row))
                noisy = train[:,i] + gauss
                train[:,i] = noisy

        elif type == 's&p':
            amount=0.04
            s_vs_p=0.5
            total_pixels = train.shape[0]
            for i in range(train.shape[1]):
                out = np.copy(train[:,i])
                num_salt = np.ceil(amount * total_pixels * s_vs_p).astype(int)
                salt_indices = np.random.randint(0, total_pixels, num_salt)
                out[salt_indices] = 1
                    # Pepper mode
                num_pepper = np.ceil(amount * total_pixels * (1 - s_vs_p)).astype(int)
                pepper_indices = np.random.randint(0, total_pixels, num_pepper)
                out[pepper_indices] = 0
                train[:,i] = out
        
        return train

    # train the network using the update functions
    def train(self, rng, images, labels, num_epochs, test_images, test_labels, learning_rate=0.01, batch_size=20, \
              algorithm='backprop', noise=1.0, noise_type=None, report=False, report_rate=10):
        """
        Trains the network with algorithm in batches for the given number of epochs on the data provided.

        Uses batches with size as indicated by batch_size and given learning rate.

        For perturbation methods, uses SD of noise as given.

        Categorization accuracy on a test set is also calculated.

        Prints a message every report_rate epochs if requested.

        Returns an array of the losses achieved at each epoch (and accuracies if test data given).
        """

        if algorithm == 'node_perturb':
            print('helloooooo, I am able to print, tada!!!')
        # provide an output message
        if report:
            print("Training starting...")

        # make batches from the data
        batches = create_batches(rng, batch_size, images.shape[1])

        # create arrays to store loss and accuracy values
        losses = np.zeros((num_epochs * batches.shape[0],))
        accuracy = np.zeros((num_epochs,))
        test_loss = np.zeros((num_epochs,))
        cosine_similarity = np.zeros((num_epochs,2))

        # estimate the gradient SNR on the test set
        grad = np.zeros((test_images.shape[1], *self.W_h_1.shape))
        for t in range(test_images.shape[1]):
            inputs = test_images[:, [t]]
            targets = test_labels[:, [t]]
            grad[t, ...], _, _ = self.return_grad(rng, inputs, targets, algorithm=algorithm, eta=0., noise=noise)
        snr = calculate_grad_snr(grad)

        if noise_type in ['gauss', 's&p']:
            inputs = self.alter_inputs(np.copy(images), noise_type)
        else: 
            inputs = images

        # run the training for the given number of epochs
        update_counter = 0
        for epoch in range(num_epochs):

            # step through each batch
            for b in range(batches.shape[0]):
                # get the inputs and targets for this batch
                batch_input = inputs[:, batches[b, :]]
                targets = labels[:, batches[b, :]]

                # calculate the current loss
                losses[update_counter] = self.mse_loss(rng, batch_input, targets)

                # update the weights
                self.update(rng, batch_input, targets, eta=learning_rate, algorithm=algorithm, noise=noise)
                update_counter += 1

            # calculate the current test accuracy
            (testhid1, testhid2, testout) = self.inference(rng, test_images)
            accuracy[epoch] = calculate_accuracy(testout, test_labels)
            test_loss[epoch] = self.mse_loss(rng, test_images, test_labels)
            hid1, hid2, _ = self.return_grad(rng, test_images, test_labels, algorithm=algorithm, eta=0., noise=noise)
            bphid1, bphid2, _ = self.return_grad(rng, test_images, test_labels, algorithm='backprop', eta=0., noise=noise)

            cos_sim_l1 = calculate_cosine_similarity(hid1, bphid1)
            cos_sim_l2 = calculate_cosine_similarity(hid2, bphid2)

            cosine_similarity[epoch, :] = [cos_sim_l1, cos_sim_l2]

            # print an output message every report_rate epochs
            if report and np.mod(epoch + 1, report_rate) == 0:
                print("...completed ", (epoch + 1)/report_rate,
                      " epochs of training. Current loss: ", round(losses[update_counter - 1], 2))

        # provide an output message
        if report:
            print("Training complete.")

        return (losses, accuracy, test_loss, snr)
    

    def train_nonstat_data(self, rng, images, labels, num_epochs, test_images, test_labels, learning_rate=0.01, batch_size=20, \
              algorithm='backprop', noise=1.0, report=False, report_rate=10):
        """
        Trains the network with algorithm in batches for the given number of epochs on the data provided.

        Uses batches with size as indicated by batch_size and given learning rate.

        For perturbation methods, uses SD of noise as given.

        Categorization accuracy on a test set is also calculated.

        Prints a message every report_rate epochs if requested.

        Returns an array of the losses achieved at each epoch (and accuracies if test data given).
        """


        '''
        ideas: 
        - give data sorted according to non-stationary distribution
        - sort data in this method and then use it accordingly
        - parameter depending on non-stationarity type
        '''


        # sort data here
        # 50/50 split
        in_first_half = [1 if sum(labels[0:4, i]) == 1 else 0 for i in range(images.shape[1])]
        images_first_half = images[:, np.where(in_first_half)[0]]
        labels_first_half = labels[:, np.where(in_first_half)[0]]
        images_second_half = images # [:, np.where(1 - np.array(in_first_half))[0]] 
        labels_second_half = labels # [:, np.where(1 - np.array(in_first_half))[0]]


        # provide an output message
        if report:
            print("Training starting...")

        # make batches from the data
        batches_1 = create_batches(rng, batch_size, images_first_half.shape[1])
        batches_2 = create_batches(rng, batch_size, images_second_half.shape[1])

        # create arrays to store loss and accuracy values
        losses = np.zeros((num_epochs * (batches_2.shape[0]),)) # size of backprop vector
        accuracy = np.zeros((num_epochs,))
        test_loss = np.zeros((num_epochs,))
        cosine_similarity = np.zeros((num_epochs,))

        # estimate the gradient SNR on the test set
        grad = np.zeros((test_images.shape[1], *self.W_h_1.shape, *self.W_h_2.shape))
        for t in range(test_images.shape[1]):
            inputs = test_images[:, [t]]
            targets = test_labels[:, [t]]
            grad[t, ...], _ = self.return_grad(rng, inputs, targets, algorithm=algorithm, eta=0., noise=noise)
        snr = calculate_grad_snr(grad)

        # run the training for the given number of epochs
        update_counter = 0
        first_half = 1
        for epoch in range(num_epochs):
            if first_half:
                images = images_first_half 
                labels = labels_first_half
                batches = batches_1
            else:
                images = images_second_half
                labels = labels_second_half
                batches = batches_2

            # step through each batch
            for b in range(batches.shape[0]):
                # get the inputs and targets for this batch
                inputs = images[:, batches[b, :]]
                targets = labels[:, batches[b, :]]

                # calculate the current loss
                losses[update_counter] = self.mse_loss(rng, inputs, targets)

                # update the weights
                self.update(rng, inputs, targets, eta=learning_rate, algorithm=algorithm, noise=noise)
                update_counter += 1

                if update_counter == losses.shape[0]:
                    break

            # calculate the current test accuracy
            (testhid, testout) = self.inference(rng, test_images)
            accuracy[epoch] = calculate_accuracy(testout, test_labels)
            test_loss[epoch] = self.mse_loss(rng, test_images, test_labels)
            grad_test, _ = self.return_grad(rng, test_images, test_labels, algorithm=algorithm, eta=0., noise=noise)
            grad_bp, _ = self.return_grad(rng, test_images, test_labels, algorithm='backprop', eta=0., noise=noise)
            cosine_similarity[epoch] = calculate_cosine_similarity(grad_test, grad_bp)

            # print an output message every 10 epochs
            if report and np.mod(epoch + 1, report_rate) == 0:
                print("...completed ", epoch + 1,
                      " epochs of training. Current loss: ", round(losses[update_counter - 1], 2))
                
            if epoch == int(num_epochs/2):
                first_half = 0


        # provide an output message
        if report:
            print("Training complete.")

        return (losses, accuracy, test_loss, snr)
    

    def train_online(self, rng, images, labels, test_images, test_labels, learning_rate=0.01, max_it=None, conv_loss = 5e-2, algorithm='backprop', noise=1.0, report=False, report_rate=100):
        """
        Trains the network with online learning algorithm.

        For perturbation methods, uses SD of noise as given.

        Categorization accuracy on a test set is also calculated.

        Prints a message every report_rate iterations if requested.

        Returns an array of the losses achieved every x number of iterations (and accuracies if test data given).
        """

        # provide an output message
        if report:
            print("Training starting...")

        losses = []
        accuracy = []
        test_loss = []
        cosine_similarity = []

        # estimate the gradient SNR on the test set
        grad = np.zeros((test_images.shape[1], *self.W_h_1.shape, *self.W_h_2.shape))
        for t in range(test_images.shape[1]):
            inputs = test_images[:, [t]]
            targets = test_labels[:, [t]]
            grad[t, ...], _ = self.return_grad(rng, inputs, targets, algorithm=algorithm, eta=0., noise=noise)
        snr = calculate_grad_snr(grad)

        converged = False
        update_counter = 0
        while not converged and (max_it is None or int(update_counter/report_rate) < max_it):
            t = rng.integers(images.shape[1]) # choose a random image
            inputs = images[:, [t]]
            targets = labels[:, [t]]

            # calculate the current loss
            loss = self.mse_loss(rng, inputs, targets)

            # store the loss every report_rate samples, should be batchsize to be comparable to other methods
            if update_counter % report_rate == 0:
                losses.append(loss)  

                # calculate the current test accuracy
                (testhid, testout) = self.inference(rng, test_images)
                accuracy.append(calculate_accuracy(testout, test_labels))
                test_loss.append(self.mse_loss(rng, test_images, test_labels))
                grad_test, _ = self.return_grad(rng, test_images, test_labels, algorithm=algorithm, eta=0., noise=noise)
                grad_bp, _ = self.return_grad(rng, test_images, test_labels, algorithm='backprop', eta=0., noise=noise)
                cosine_similarity.append(calculate_cosine_similarity(grad_test, grad_bp))

            # print an output message every 100 iterations
            if report and np.mod(update_counter+1, images.shape[1]) == 0: # report_rate*10
                print("...completed ", (update_counter + 1),
                        " iterations (corresponding to 1 epoch) of training data (single images). Current loss: ", round(losses[-1], 4), ".")

            # check for convergence
            if loss < conv_loss:
                converged = True
                losses.append(loss)  
                print("...completed ", (update_counter + 1),
                        " iterations of training data (single images). Current loss: ", round(losses[-1], 4))

            # update the weights
            self.update(rng, inputs, targets, eta=learning_rate, algorithm=algorithm, noise=noise)
            update_counter += 1      
        

        # provide an output message
        if report:
            print("Training complete.")

        return (losses, accuracy, test_loss, snr)
    
class NodePerturbMLP(MLP):
    """
    A multilayer perceptron that is capable of learning through node perturbation
    """

    def node_perturb(self, rng, inputs, targets, noise=1.0):
        """
        Calculates the weight updates for node perturbation learning, using noise with SD as given
        """

        # get the random perturbations
        hidden1, hidden2, output = self.inference(rng, inputs)
        hidden1_p, hidden2_p, output_p = self.inference(rng, inputs, noise=noise)

        loss_now = self.mse_loss_batch(rng, inputs, targets, output=output)
        loss_per = self.mse_loss_batch(rng, inputs, targets, output=output_p)
        delta_loss = loss_now - loss_per

        hidden1_update = np.mean(
            delta_loss * (((hidden1_p - hidden1) / noise ** 2)[:, None, :] * add_bias(inputs)[None, :, :]), axis=2)
        
        hidden2_update = np.mean(
            delta_loss * (((hidden2_p - hidden2) / noise ** 2)[:, None, :] * add_bias(hidden1)[None, :, :]), axis=2)
        
        output_update = np.mean(
            delta_loss * (((output_p - output) / noise ** 2)[:, None, :] * add_bias(hidden2_p)[None, :, :]), axis=2)

        return (hidden1_update, hidden2_update, output_update)
    
class KolenPollackMLP(MLP):
    """
    A multilayer perceptron that is capable of learning through the Kolen-Pollack algorithm
    """

    def kolepoll(self, rng, inputs, targets, eta_back=0.01):
        """
        Calculates the weight updates for Kolen-Polack learning
        """
        ###################################################################
        ## Fill out the following then remove
        #raise NotImplementedError("Student exercise: calculate updates.")
        ###################################################################

        # do a forward pass
        (hidden1, hidden2, output) = self.inference(rng, inputs)

        # calculate the updates for the forward weights
        error = targets - output
        delta_W_h_1 = np.dot(np.dot(self.B, error * self.act_deriv(output)) * self.act_deriv(hidden1), \
                           add_bias(inputs).transpose())
        delta_W_h_2 = np.dot(np.dot(self.B, error * self.act_deriv(output)) * self.act_deriv(hidden2), \
                           add_bias(hidden1).transpose())
        
        #delta_err = np.dot(self.W_y.T, error)
        delta_err = np.dot(error * self.act_deriv(output), add_bias(hidden2).transpose())
        delta_W_y = delta_err - 0.1 * self.W_y

        # calculate the updates for the backwards weights and implement them
        delta_B = delta_err[:, :-1].transpose() - 0.1 * self.B
        self.B += eta_back * delta_B
        return (delta_W_h_1, delta_W_h_2, delta_W_y)