import matplotlib.pyplot as plt

# Data
batches = [1, 8, 16, 64]
epochs_1 = [72.5, 65.4, 61.1, 52.3]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(batches, epochs_1, marker='o', label='1024 Records')

plt.title('Reconstruction Accuracy for different No. of batches')
plt.xlabel('No. of batches')
plt.ylabel('Reconstruction Accuracy (%)')
plt.xticks(batches)
plt.legend()
plt.grid(True)
plt.show()
