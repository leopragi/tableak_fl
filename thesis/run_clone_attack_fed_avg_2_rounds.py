import torch
import numpy as np
import copy
from attacks import invert_grad
from models import FullyConnected
from datasets import ADULT
from utils import get_acc_and_bac
from attacks import fed_avg_attack
from utils import match_reconstruction_ground_truth
import csv
import os

nodes = [4]
batch_sizes = [16] 
# rounds = [1, 2, 3, 4, 5, 6] 

architecture_layout = [100, 100, 2]  # network architecture (fully connected)
n_local_epochs = 1
lr = 0.01
post_selection = 15
criterion = torch.nn.CrossEntropyLoss()


if not os.path.exists('csv'):
   os.makedirs('csv')

dataset = ADULT()
dataset.standardize()
Xtrain, ytrain = dataset.get_Xtrain(), dataset.get_ytrain()

for n_clients in nodes:
    for batch_size in batch_sizes:
        # list to store reconstructed x tensors
        # rec_x_s = None
        # true_y_s = None
        uid = f'{n_clients}_{batch_size}'
        reconstructed_file_name = f'csv/{uid}_reconstructed.csv'
        ground_truth_file_name = f'csv/{uid}_ground_truth.csv'
        accuracy_file_name = f'csv/{uid}_accuracy.csv'
        print(f'Starting {uid}...')

        print(f'Creating csv files...')
        file = open(reconstructed_file_name, 'w')
        writer = csv.writer(file)
        writer.writerows([dataset.features.keys()])
        file.close()

        file = open(ground_truth_file_name, 'w')
        writer = csv.writer(file)
        writer.writerows([dataset.features.keys()])
        file.close()

        file = open(accuracy_file_name, 'w')
        writer = csv.writer(file)
        file.close()

        # now, instantiate a global neural network, that serves as initial copy of all client neural networks
        global_net = FullyConnected(dataset.num_features, architecture_layout)
        client_nets = [copy.deepcopy(global_net) for _ in range(n_clients)]

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        last_params = global_net.parameters()

        data_list = [batch for batch in trainloader]

        round = 2
        node = 0
        i = 0

        while i < len(data_list) - round:
            client_id = node % 4
            client_net = client_nets[client_id]

            for _ in range(round):
                i = i + 1
                data_x, data_y = data_list[i]

                for last, client in zip(last_params, client_net.parameters()):
                    client.data.copy_(last.data)

                    outputs = client_net(data_x)
                    loss = criterion(outputs, data_y)

                    grad = torch.autograd.grad(loss, client_net.parameters(), retain_graph=True)

                    with torch.no_grad():
                        for param, param_grad in zip(client_net.parameters(), grad):
                            param -= lr * param_grad

                    last_params = client_net.parameters()

            if(client_id == 1):
                prev_x, prev_y = data_list[i - 1]
                for n in range(2, n_clients * round):
                    if(i - n < 0): break
                    l_prev_x, l_prev_y = data_list[i - n]
                    prev_x = torch.cat((prev_x, l_prev_x), 0)
                    prev_y = torch.cat((prev_y, l_prev_y), 0)

                per_client_ground_truth_data = [prev_x.detach().clone()]
                per_client_ground_truth_labels = [prev_y.detach().clone()]
                attacked_clients_params = [[param.clone().detach() for param in last_params]]

                per_client_best_scores = None
                per_client_best_reconstructions = None

                for _ in range(post_selection):
                    per_client_candidate_reconstructions, per_client_final_losses = fed_avg_attack(
                        original_net=copy.deepcopy(client_net),
                        attacked_clients_params=attacked_clients_params,
                        attack_iterations=1500,
                        attack_learning_rate=0.06,
                        n_local_epochs=n_local_epochs,
                        local_batch_size=batch_size,
                        lr=lr,
                        dataset=dataset,
                        per_client_ground_truth_data=per_client_ground_truth_data,
                        per_client_ground_truth_labels=per_client_ground_truth_labels,
                        reconstruction_loss='cosine_sim',
                        priors=None,
                        # epoch_matching_prior=(0.001, 'mean_squared_error'),
                        initialization_mode='uniform',
                        softmax_trick=False,
                        gumbel_softmax_trick=False,
                        sigmoid_trick=False,
                        temperature_mode='constant',
                        sign_trick=True,
                        apply_projection_to_features=None,
                        device=None
                    )

                    if per_client_best_scores is None or per_client_best_scores > per_client_final_losses[0]:
                        per_client_best_scores = per_client_final_losses[0]
                        per_client_best_reconstructions = per_client_candidate_reconstructions[0].detach().clone()

                rec_x_mixed, true_x_mixed = dataset.decode_batch(per_client_best_reconstructions, standardized=True), dataset.decode_batch(prev_x.detach(), standardized=True)

                # if rec_x_s == None:
                #     rec_x_s = rec_x_mixed
                #     true_y_s = per_client_ground_truth_labels[0]
                # else:
                #     rec_x_s = torch.cat([rec_x_s, rec_x_mixed])
                #     true_y_s = torch.cat([true_y_s, per_client_ground_truth_labels[0]])

                with open(reconstructed_file_name, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(np.column_stack([rec_x_mixed, per_client_ground_truth_labels[0]]))

                with open(ground_truth_file_name, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(np.column_stack([true_x_mixed, per_client_ground_truth_labels[0]]))

                tolerance_map = dataset.create_tolerance_map()
                _, error_map, _, _ = match_reconstruction_ground_truth(true_x_mixed, rec_x_mixed, tolerance_map)
                reconstruction_accuracy = 100 * (1 - np.mean(error_map))

                with open(accuracy_file_name, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(map(lambda x: [x], [reconstruction_accuracy]))

            node = node + 1

        print(f'Done....')