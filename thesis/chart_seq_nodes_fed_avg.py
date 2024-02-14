import matplotlib.pyplot as plt

# No of nodes: 2
nodes_2_reconstruction_accuracy = [88.0, 95.0, 95.0, 100.0, 95.0, 99.0, 96.0, 96.0, 94.0, 68.0, 99.0, 87.0, 100.0, 94.0, 87.0]
nodes_2_model_accuracy = [27.0, 27.0, 28.0, 29.0, 31.0, 31.0, 33.0, 35.0, 38.0, 41.0, 45.0, 48.0, 47.0, 50.0, 53.0]

# No of nodes: 4
nodes_4_reconstruction_accuracy = [98.0, 74.0, 79.0, 86.0, 79.0, 86.0, 82.0, 74.0, 80.0, 95.0, 85.0, 73.0, 82.0, 83.0, 75.0]
nodes_4_model_accuracy = [68.0, 72.0, 73.0, 74.0, 74.0, 74.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0]

# No of nodes: 8
nodes_8_reconstruction_accuracy = [100.0, 64.0, 77.0, 74.0, 75.0, 72.0, 83.0, 72.0, 73.0, 80.0, 79.0, 80.0, 74.0, 77.0, 76.0]
nodes_8_model_accuracy = [73.0, 74.0, 74.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0]

# No of nodes: 16
nodes_16_reconstruction_accuracy = [100.0, 83.0, 76.0, 69.0, 73.0, 70.0, 72.0, 79.0, 88.0, 78.0, 70.0, 80.0, 79.0, 71.0, 79.0]
nodes_16_model_accuracy = [58.0, 63.0, 67.0, 70.0, 71.0, 70.0, 67.0, 70.0, 71.0, 73.0, 74.0, 74.0, 74.0, 73.0, 66.0]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 6))

# Subplot 1
axs[0, 0].plot(nodes_2_reconstruction_accuracy, label='Nodes: 2 - Reconstruction')
axs[0, 0].plot(nodes_2_model_accuracy, label='Nodes: 2 - Model')
axs[0, 0].set_title('Nodes: 2')
axs[0, 0].legend()

# Subplot 2
axs[0, 1].plot(nodes_4_reconstruction_accuracy, label='Nodes: 4 - Reconstruction')
axs[0, 1].plot(nodes_4_model_accuracy, label='Nodes: 4 - Model')
axs[0, 1].set_title('Nodes: 4')
axs[0, 1].legend()

# Subplot 3
axs[1, 0].plot(nodes_8_reconstruction_accuracy, label='Nodes: 8 - Reconstruction')
axs[1, 0].plot(nodes_8_model_accuracy, label='Nodes: 8 - Model')
axs[1, 0].set_title('Nodes: 8')
axs[1, 0].legend()

# Subplot 4
axs[1, 1].plot(nodes_16_reconstruction_accuracy, label='Nodes: 16 - Reconstruction')
axs[1, 1].plot(nodes_16_model_accuracy, label='Nodes: 16 - Model')
axs[1, 1].set_title('Nodes: 16')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()
plt.show()