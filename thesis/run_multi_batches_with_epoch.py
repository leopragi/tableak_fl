import torch
import copy
import numpy as np
from attacks import invert_grad
from models import FullyConnected, FullyConnectedTrainer
from datasets import ADULT
from utils import match_reconstruction_ground_truth
from torch import nn as nn


print('Instantiate the dataset')
dataset = ADULT()
dataset.standardize()
Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()

expriments = [
    # [64, 1],
    # [64, 3],
    # [64, 5],
    # [128, 1],
    # [128, 3],
    # [128, 5],
    [1024, 1],
    [1024, 5],
    [1024, 10],
]

for expriment in expriments:

    # sample a random batch we are going to invert
    batch_size, epoch = expriment

    random_indices = np.random.randint(0, len(Xtrain), int(batch_size)) 
    true_x, true_y = Xtrain[random_indices], ytrain[random_indices]

    # different batch splits (1 batch, 8 batches, 16 batches, 64 batches)
    batch_size_1 = batch_size
    batch_size_8 = batch_size / 8
    batch_size_16 = batch_size / 16
    batch_size_64 = batch_size / 64

    for l_batch_size in [batch_size_1, batch_size_8, batch_size_16, batch_size_64]:

        # now, instantiate 2 neural network, one for whole batch and other for batch-by-batch
        original_net = FullyConnected(Xtrain.size()[1], [100, 100, 2])
        trained_net = FullyConnected(Xtrain.size()[1], [100, 100, 2])

        trained_net = copy.deepcopy(original_net)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(trained_net.parameters())
        trainer = FullyConnectedTrainer(data_x=true_x.detach().clone(), data_y=true_y.detach().clone(),
                                        optimizer=optimizer, criterion=criterion, verbose=False)
        # collect the averaged gradients
        avg_grads = trainer.train(trained_net, epoch, l_batch_size, shuffle=False, reset=False)

        # now we have obtained the true gradient that is shared with the server, and can simulate the attack from the server's side
        rec_x = invert_grad(
            net=original_net, 
            training_criterion=criterion,
            true_grad=avg_grads,
            true_label=true_y,  # note that we assume knoweldge of the labels
            true_data=true_x,  # only used for shaping, not used in the actual inversion
            reconstruction_loss='cosine_sim',
            dataset=dataset,
            max_iterations=1500,
            # the following parameter setup below corresponds to TabLeak as in the paper
            post_selection=30,
            softmax_trick=True,
            sigmoid_trick=True,
            pooling='median+softmax',
            verbose=False
        )

        # rec_x is the reconstruction, but still standardized and one-hot encoded
        # to evaluate it, we project both the true data and the reconsutruction back to mixed representation
        true_x_mixed, rec_x_mixed = dataset.decode_batch(true_x, standardized=True), dataset.decode_batch(rec_x.detach(), standardized=True)

        # now we match the rows of the two batches and obtain an error map, the average of which is the error of the reconstruction
        tolerance_map = dataset.create_tolerance_map()
        _, error_map, _, _ = match_reconstruction_ground_truth(true_x_mixed, rec_x_mixed, tolerance_map)
        reconstruction_accuracy = 100 * (1 - np.mean(error_map))

        print(f'Batch_size={l_batch_size} Epochs={epoch}: Reconstruction_accuracy={reconstruction_accuracy:.1f}%')