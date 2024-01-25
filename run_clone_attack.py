import torch
import copy
import numpy as np
from attacks import invert_grad
from models import FullyConnected, FullyConnectedTrainer
from datasets import ADULT, Lawschool
from utils import match_reconstruction_ground_truth
from torch import nn as nn
from utils import get_acc_and_bac
import csv


reconstructed_file_name = 'reconstructed.csv'
ground_truth_file_name = 'ground_truth.csv'

print(f'Creating csv files...')
file = open(reconstructed_file_name, 'w')
file.close()
file = open(ground_truth_file_name, 'w')
file.close()

print('Instantiate the dataset')
dataset = ADULT()
dataset.standardize()

batch_size = 32

Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()
testx, testy = dataset.get_Xtest(), dataset.get_ytest()

# now, instantiate 2 neural network, one for whole batch and other for batch-by-batch
original_net = FullyConnected(Xtrain.size()[1], [100, 100, 2])
attack_net = FullyConnected(Xtrain.size()[1], [100, 100, 2])

# list to store reconstructed x tensors
rec_x_s = None
true_y_s = None

criterion = torch.nn.CrossEntropyLoss()
original_optimizer = torch.optim.Adam(original_net.parameters())

# split into batches to train batch-by-batch
n = int(np.ceil(Xtrain.size()[0] / batch_size))

for i in range(n):
    # copy the original net to do the attack later for every round
    attack_net = copy.deepcopy(original_net)

    upper_line = min((i + 1) * batch_size, Xtrain.size()[0])
    bottom_line = i * batch_size

    true_x = Xtrain[bottom_line:upper_line]
    true_y = ytrain[bottom_line:upper_line]

    original_optimizer.zero_grad()
    outputs = original_net(true_x)
    loss = criterion(outputs, true_y)
    loss.backward()
    original_optimizer.step()

    true_grads = [param.grad.detach().clone() for param in original_net.parameters()]

    rec_x = invert_grad(
        net=attack_net, 
        training_criterion=criterion,
        true_grad=true_grads,
        true_label=true_y,  # note that we assume knoweldge of the labels
        true_data=true_x,  # only used for shaping, not used in the actual inversion
        reconstruction_loss='cosine_sim',
        dataset=dataset,
        max_iterations=20,
        # the following parameter setup below corresponds to TabLeak as in the paper
        post_selection=15,
        softmax_trick=True,
        sigmoid_trick=True,
        pooling='median+softmax',
        verbose=False
    )

    # rec_x is the reconstruction, but still standardized and one-hot encoded
    # to evaluate it, we project both the true data and the reconsutruction back to mixed representation
    true_x_mixed, rec_x_mixed = dataset.decode_batch(true_x, standardized=True), dataset.decode_batch(rec_x.detach(), standardized=True)

    if rec_x_s == None:
        rec_x_s = rec_x
        true_y_s = true_y
    else:
        rec_x_s = torch.cat([rec_x_s, rec_x])
        true_y_s = torch.cat([true_y_s, true_y])

    with open(reconstructed_file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.column_stack([rec_x_mixed, true_y]))

    with open(ground_truth_file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np.column_stack([true_x_mixed, true_y]))

    # now we match the rows of the two batches and obtain an error map, the average of which is the error of the reconstruction
    tolerance_map = dataset.create_tolerance_map()
    _, error_map, _, _ = match_reconstruction_ground_truth(true_x_mixed, rec_x_mixed, tolerance_map)
    reconstruction_accuracy = 100 * (1 - np.mean(error_map))

    # get the accuracy of model
    acc, _ = get_acc_and_bac(original_net, testx, testy)
    print(f"Round: {i + 1}/{n}, Original net accuracy: {(acc * 100):.1f}, Reconstruction accuracy: {reconstruction_accuracy}")

    if i > 10: break

# attacker net - to train the reconstrued batch
clone_net = FullyConnected(Xtrain.size()[1], [100, 100, 2])
acc, bac = get_acc_and_bac(clone_net, testx, testy)
print(f'Before training: Clone net accuracy: {acc * 100:.1f}%')
optimizer = torch.optim.Adam(clone_net.parameters())
trainer = FullyConnectedTrainer(
    data_x=rec_x_s, 
    data_y=true_y_s,
    optimizer=optimizer, 
    criterion=criterion, 
    verbose=False
)
trainer.train(clone_net, 1, batch_size)
acc, bac = get_acc_and_bac(clone_net, testx, testy)
print(f'After training: Clone net accuracy: {acc * 100:.1f}%')