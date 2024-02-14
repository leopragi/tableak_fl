import matplotlib.pyplot as plt

# Data
batches = [1, 8, 16, 64]
epochs_1 = [72.5, 65.4, 61.1, 52.3]
epochs_5 = [67.5, 58.1, 53.9, 46.6]
epochs_10 = [64.9, 51.3, 48.4, 45.5]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(batches, epochs_1, marker='o', label='Epochs=1')
plt.plot(batches, epochs_5, marker='o', label='Epochs=5')
plt.plot(batches, epochs_10, marker='o', label='Epochs=10')

plt.title('Reconstruction Accuracy for different No. of batches and epochs')
plt.xlabel('No. of batches')
plt.ylabel('Reconstruction Accuracy (%)')
plt.xticks(batches)
plt.legend()
plt.grid(True)
plt.show()
