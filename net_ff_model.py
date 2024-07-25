'''
Functions: 
- The file provide useful functions to train and test the forward-forward methods.
'''

import hydra
import numpy as np
import torch
import torch.nn as nn
import torchvision
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import helpers
from forward_forward_complex.src import ff_mnist
from forward_forward_complex.src import ff_model as ff_com_model
from forward_forward_complex.src import utils
from MLP import MLP

torch.utils.backcompat.broadcast_warning.enabled = False
torch.utils.backcompat.keepdim_warning.enabled = False

img_size = 28
num_classes = 10

def datatype_conversion(images, targets, batches, index, mode='train'):
    if mode == 'train':
        inputs = torch.from_numpy(images[:, batches[index, :]].transpose()).float()
        labels = torch.from_numpy(targets[:, batches[index, :]].transpose())
        labels = torch.argmax(labels, dim=1)
    else:
        inputs = torch.from_numpy(images.transpose()).float()
        labels = torch.from_numpy(targets.transpose())
        labels = torch.argmax(labels, dim=1)
    return inputs, labels
    
    
def load_config():
    initialize(config_path="./forward_forward_complex", 
               version_base=None)
    cfg = compose(config_name="config")
    return cfg

class net_FF_model:
    
    '''
    
    - Training Loss
    - Testing Loss
    - Validation Accuracy
    
    '''
    
    def __init__(self, rng, mlp_bp=None, lr=1e-3):
        """
        Function: 
            - Initialize the FF_model class.
        Args:
            - mlp_bp (MLP): Default MLP model with Backpropagation.
            - lr (Float, optional): Learning Rate. Defaults to 1e-3.
        """
        ### Begin: Fundamental Initialization ###
        self.opt = load_config()
        self.rng = rng
        self.mlp = {}
        self.mlp_opt = {}
        self.loss_fn = {
            'mse': nn.MSELoss(),
            'cross_entropy': nn.CrossEntropyLoss()
        }
        ### End: Fundamental Initialization   ###
        
        
        ### Begin: MLP with Backpropagation Initialization ###
        self.mlp['bp'] = mlp_bp
        ### End: MLP with Backpropagation Initialization   ###
        
        
        ### Begin: MLP with Complex Forward-Forward Initialization ###
        self.opt = utils.parse_args(self.opt)
        self.mlp['ff_com'], self.mlp_opt['ff_com'] = utils.get_model_and_optimizer(self.opt)
        ### End: MLP with Complex Forward-Forward Initialization   ###
        
        '''
        ### Begin: MLP with Simple Forward-Forward Initialization ###
        self.mlp['ff_sim'] = ff_sim_model([784, 500, 500])
        self.mlp_opt['ff_sim'] = torch.optim.SGD(self.mlp['ff_sim'].parameters(), lr=lr,
                                                 momentum=0., weight_decay=0., dampening=0., nesterov=False)
        ### End: MLP with Simple Forward-Forward Initialization   ###
        '''
           
    def train_over(self, train_images, train_targets, 
                   test_images, test_targets,
                   batch_size=128, epochs=25, 
                   model='ff_com', return_loss='cross_entropy'):

        running_real_losses = []
        running_return_losses = []
        real_losses = []
        return_losses = []
        test_losses = []
        test_accuracies = []
        
        batches = helpers.create_batches(self.rng, batch_size, train_images.shape[1])
        
        self.mlp[model].train()
        for epoch in range(epochs):
            
            # training ...
            for b in range(batches.shape[0]):
                inputs, labels = datatype_conversion(train_images, train_targets, batches, b, mode='train')
                inputs, labels = self._input_conversion(inputs, labels, model)
                
                running_real_losses.append(
                    self.train_batch(inputs, labels, model)
                )
                running_return_losses.append(
                    self._get_inference_loss(inputs, labels, model, return_loss)
                )
            real_losses.append(np.mean(running_real_losses))
            return_losses.append(np.mean(running_return_losses))

            # accuracy and loss in the testing data
            test_accuracy, test_loss = self.test(test_images, test_targets, model, return_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            
            # report
            print(f'''...completed  {epoch}  epochs of training. \n
            Current innate loss:  {real_losses[epoch]}, \n
            Current training {return_loss} loss:  {return_losses[epoch]}, \n
            Current testing {return_loss} loss:  {test_losses[epoch]}, \n   
            Current testing accuracy:  {test_accuracies[epoch]} \n         
            ''')
            
        return return_losses, test_losses, test_accuracies

    def train_online(self, train_images, train_targets, 
                   test_images, test_targets,
                   max_it=10000, report_rate=100, conv_loss=1e-2, lr=1e-2,
                   model='ff_com', return_loss='cross_entropy'):
        
        for param_group in self.mlp_opt[model].param_groups:
            param_group['lr'] = lr
        
        running_real_losses = []
        running_return_losses = []
        return_losses = []
        real_losses = [] 
        test_losses = []
        test_accuracies = []
        
        self.mlp[model].train()
        converged = False
        update_counter = 0
        train_images, train_targets = datatype_conversion(
            train_images, train_targets, None, None, mode='online'
        )
        
        while not converged and update_counter < max_it:
            # training ...
            t = self.rng.integers(train_images.shape[0])
            inputs, labels = self._input_conversion(
                train_images[t].unsqueeze(0), 
                train_targets[t].unsqueeze(0), 
                model
            )
            
            running_real_losses.append(
                self.train_batch(inputs, labels, model)
            )
            running_return_losses.append(
                self._get_inference_loss(inputs, labels, model, return_loss)
            )

            # accuracy and loss in the testing data
            if update_counter % (report_rate) == 0:
                real_losses.append(sum(running_real_losses) / len(running_real_losses))
                return_losses.append(sum(running_return_losses) / len(running_return_losses))
                running_real_losses = []
                running_return_losses = []
                
                test_accuracy, test_loss = self.test(test_images, test_targets, model, return_loss)
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)



            if update_counter % (report_rate * 10) == 0:
                # report
                print(f'''...completed  {update_counter + 1}  iterations of training. \n
                Current testing accuracy:  {np.mean(test_accuracies[-report_rate : -1])} \n
                Current testing loss:  {np.mean(test_losses[-report_rate : -1])} \n ''')

            if sum(return_losses) < conv_loss:
                converged = True
                # report
                print(f'''...Converge at {update_counter + 1} iterations of training.''')
            
            update_counter += 1
            
        return return_losses, test_losses, test_accuracies
    
    def train_nonstationary(self, train_images, train_targets, 
                   test_images, test_targets,
                   batch_size=128, epochs=25, 
                   model='ff_com', return_loss='cross_entropy'):
        # sort data here
        # 50/50 split
        in_first_half = [1 if sum(train_targets[0:4, i]) == 1 else 0 for i in range(train_images.shape[1])]
        images_first_half = train_images[:, np.where(in_first_half)[0]]
        labels_first_half = train_targets[:, np.where(in_first_half)[0]]
        images_second_half = train_images[:, np.where(1 - np.array(in_first_half))[0]] # unsure about this notation
        labels_second_half = train_targets[:, np.where(1 - np.array(in_first_half))[0]]

        # prepare for training
        running_real_losses = []
        running_return_losses = []
        real_losses = []
        return_losses = []
        test_losses = []
        test_accuracies = []
        first_half = 1
        
        # make batches from the data
        batches_1 = helpers.create_batches(self.rng, batch_size, images_first_half.shape[1])
        batches_2 = helpers.create_batches(self.rng, batch_size, images_second_half.shape[1])
        
        # start training
        self.mlp[model].train()
        for epoch in range(epochs):
            if first_half:
                images = images_first_half 
                targets = labels_first_half
                batches = batches_1
            else:
                images = images_second_half
                targets = labels_second_half
                batches = batches_2
            for b in range(batches.shape[0]):
                inputs, labels = datatype_conversion(images, targets, batches, b, mode='train')
                inputs, labels = self._input_conversion(inputs, labels, model)
                
                running_real_losses.append(
                    self.train_batch(inputs, labels, model)
                )
                running_return_losses.append(
                    self._get_inference_loss(inputs, labels, model, return_loss)
                )
            real_losses.append(np.mean(running_real_losses))
            return_losses.append(np.mean(running_return_losses))

            # accuracy and loss in the testing data
            test_accuracy, test_loss = self.test(test_images, test_targets, model, return_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            
            # report
            print(f'''...completed  {epoch}  epochs of training. \n
            Current innate loss:  {real_losses[epoch]}, \n
            Current training {return_loss} loss:  {return_losses[epoch]}, \n
            Current testing {return_loss} loss:  {test_losses[epoch]}, \n   
            Current testing accuracy:  {test_accuracies[epoch]} \n         
            ''')
            
            if epoch == int(epochs // 2):
                first_half = 0
            
        return return_losses, test_losses, test_accuracies
           
    def train_noisydata(self, train_images, train_targets, 
                   test_images, test_targets,
                   batch_size=128, epochs=25, 
                   model='ff_com', return_loss='cross_entropy',
                   noise_type='gauss'):

        if noise_type in ['gauss', 's&p']:
            inputs = self.alter_inputs(np.copy(train_images), noise_type)
        
        running_real_losses = []
        running_return_losses = []
        real_losses = []
        return_losses = []
        test_losses = []
        test_accuracies = []
        
        batches = helpers.create_batches(self.rng, batch_size, train_images.shape[1])
        
        self.mlp[model].train()
        for epoch in range(epochs):
            
            # training ...
            for b in range(batches.shape[0]):
                inputs, labels = datatype_conversion(train_images, train_targets, batches, b, mode='train')
                inputs, labels = self._input_conversion(inputs, labels, model)
                
                running_real_losses.append(
                    self.train_batch(inputs, labels, model)
                )
                running_return_losses.append(
                    self._get_inference_loss(inputs, labels, model, return_loss)
                )
            real_losses.append(np.mean(running_real_losses))
            return_losses.append(np.mean(running_return_losses))

            # accuracy and loss in the testing data
            test_accuracy, test_loss = self.test(test_images, test_targets, model, return_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            
            # report
            print(f'''...completed  {epoch}  epochs of training. \n
            Current innate loss:  {real_losses[epoch]}, \n
            Current training {return_loss} loss:  {return_losses[epoch]}, \n
            Current testing {return_loss} loss:  {test_losses[epoch]}, \n   
            Current testing accuracy:  {test_accuracies[epoch]} \n         
            ''')
            
        return return_losses, test_losses, test_accuracies

    
    def train_batch(self, inputs, labels, model='ff_com', return_loss='cross_entropy'):
        self.mlp_opt[model].zero_grad()

        real_loss = self._get_train_loss(inputs, labels, model)
        real_loss.backward()
        
        self.mlp_opt[model].step()
        
        return real_loss.item()
    
    
    def test(self, test_images, test_labels, model='ff_com', return_loss='cross_entropy'):
        correct_samples = 0
        total_samples = 0
        accuracy = 0
        
        testing_loss = 0
        
        self.mlp[model].eval()
        with torch.no_grad():
            inputs, labels = datatype_conversion(test_images, test_labels, None, None, mode='test')
            inputs, labels = self._input_conversion(inputs, labels, model)
            
            pred_max = self.inference_max(inputs, labels, model)
            correct_samples = torch.sum(pred_max == labels['class_labels']).item()
            total_samples += inputs['neutral_sample'].shape[0]
            accuracy = correct_samples / total_samples

            testing_loss = self._get_inference_loss(inputs, labels, model, return_loss)
        return accuracy, testing_loss
        


    def get_gradient(self, model='simple'):
        '''
        
        Returns:
            - gradient
        '''
    
    
    
    def inference_ori(self, inputs, labels, model='ff_sim'):
        if model == 'ff_com':
            scalar_outputs = self.mlp[model].forward_downstream_classification_model(
                inputs, labels
            )
            return scalar_outputs['prediction'].detach()
        elif model == 'ff_sim':
            pass  
    
    def inference_max(self, inputs, labels, model='ff_sim'):
        if model == 'ff_com':
            scalar_outputs = self.mlp[model].forward_downstream_classification_model(
                inputs, labels
            )
            pred_ori = scalar_outputs['prediction'].detach()
            pred_max = torch.argmax(pred_ori, dim=1)
            return pred_max
            
        elif model == 'ff_sim':
            pass  
    
    def _get_inference_loss(self, inputs, labels, 
                            model='ff_sim', return_loss='cross_entropy'):
        if model == 'ff_com':
            preds_ori = self.inference_ori(inputs, labels, model)
            if return_loss == 'mse':
                preds_ori = torch.nn.functional.softmax(preds_ori, dim=1)
            return self.loss_fn[return_loss](preds_ori, labels["onehot_labels"].float()).item()
        elif model == 'ff_sim':
            pass
        
        
    def _get_train_loss(self, inputs, labels, model='ff_com'):
        if model == 'ff_com':
            scalar_outputs = self.mlp[model](inputs, labels)
            return scalar_outputs["Loss"]
        elif model == 'ff_sim':
            pass          
    
    def _input_conversion(self, sample, class_label, model='ff_com'):
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        onehot_labels = torch.nn.functional.one_hot(
            class_label, num_classes=num_classes
        )
        if model == 'ff_com':
            inputs = {
                "pos_images": pos_sample,
                "neg_images": neg_sample,
                "neutral_sample": neutral_sample,
            }
            labels = {
                "class_labels": class_label,
                "onehot_labels": onehot_labels
            }
        
        return inputs, labels
        
        
    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label, dtype=torch.int64), 
            num_classes=num_classes
        )
        pos_sample = sample.clone()
        pos_sample[:, 0 : num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        batch_size = sample.size(0)
        
        possible_classes = torch.arange(num_classes).unsqueeze(0).repeat(batch_size, 1)
        mask = possible_classes != class_label.unsqueeze(1)
        possible_classes = possible_classes[mask].view(batch_size, num_classes - 1)
        wrong_class_indices = torch.randint(0, num_classes - 1, (batch_size,))
        wrong_class_labels = possible_classes[torch.arange(batch_size), wrong_class_indices]
        one_hot_labels = torch.nn.functional.one_hot(wrong_class_labels, num_classes=num_classes)
        neg_sample = sample.clone()
        neg_sample[:, :num_classes] = one_hot_labels
        
        return neg_sample

    def _get_neutral_sample(self, z):
        uniform_label = torch.ones(num_classes) / num_classes
        z[:, 0 : num_classes] = uniform_label
        return z

    def alter_inputs(self, data, type=None):
        if type == "gauss":
            row = len(data)
            mean = 0
            var = 0.1
            sigma = var**0.5
            for i in range(data.shape[1]):
                gauss = np.random.normal(mean,sigma,(row))
                noisy = data[:,i] + gauss
                data[:,i] = noisy

        elif type == 's&p':
            amount=0.04
            s_vs_p=0.5
            total_pixels = data.shape[0]
            for i in range(data.shape[1]):
                out = np.copy(data[:,i])
                num_salt = np.ceil(amount * total_pixels * s_vs_p).astype(int)
                salt_indices = np.random.randint(0, total_pixels, num_salt)
                out[salt_indices] = 1
                    # Pepper mode
                num_pepper = np.ceil(amount * total_pixels * (1 - s_vs_p)).astype(int)
                pepper_indices = np.random.randint(0, total_pixels, num_pepper)
                out[pepper_indices] = 0
                data[:,i] = out
        
        return data
    
    