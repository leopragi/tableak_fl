import matplotlib.pyplot as plt

# Data
categories = [
    4, 8, 16, 32, 64, 128
]

# Reconstruction accuracy for ADULT
adult_data = [100.0, 100.0, 91.5, 81.5, 75.4, 73.5]

# Reconstruction accuracy for ADULT_BINARY
adult_binary_data =  [54.6, 48.2, 42.9, 43.3, 50.1, 50.7]

# Reconstruction accuracy for Lawschool
lawschool_data = [100.0, 100.0, 95.5, 87.5, 79.5, 78.9]

# Reconstruction accuracy for LawschoolBinary
lawschool_binary_data = [39.3, 46.4, 47.3, 52.2, 55.1, 55.6]

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(categories, adult_data, marker='o', label='ADULT')
plt.plot(categories, adult_binary_data, marker='o', label='ADULT (Binary)')
plt.plot(categories, lawschool_data, marker='o', label='Lawschool')
plt.plot(categories, lawschool_binary_data, marker='o', label='Lawschool (Binary)')

plt.title('Reconstruction Accuracy for Different Datasets and Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Reconstruction Accuracy (%)')
plt.legend()
plt.grid(True)
plt.show()
