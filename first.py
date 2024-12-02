import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

#
x = torch.rand(5, 3)
print(x)
#
# data = [[1, 2],[3, 4]]
# x_data = torch.tensor(data)
# print(x_data)
#

# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
#
# print(f"Numpy np_array value: \n {np_array} \n")
# print(f"Tensor x_np value: \n {x_np} \n")
#
# np.multiply(np_array, 2, out=np_array)
#
# print(f"Numpy np_array after * 2 operation: \n {np_array} \n")
# print(f"Tensor x_np value after modifying numpy array: \n {x_np} \n")
#
# x_ones = torch.ones_like(x_data) # retains the properties of x_data
# print(f"Ones Tensor: \n {x_ones} \n")
#
# x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
# print(f"Random Tensor: \n {x_rand} \n")
#
# tensor = torch.rand(3,4)
#
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")
#
# # to move to cuda gpu
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')
#   print("Available")
# print(f"Device tensor is now stored on: {tensor.device}")
#
# # define with dimensions, call print in the order of
# # row, column.
# tensor = torch.ones(4, 4)
# print('First row: ',tensor[0])
# print('First column: ', tensor[:, 0])
# print('Last column:', tensor[..., -1])
# tensor[:,1] = 0
# print(tensor)
#
# #adds 5 to all values
# print(tensor, "\n")
# tensor.add_(5)
#
# print("after add opp \n", tensor)

# when defining a tensor at the location of a numpy
# array, changes to the numpy causes a change to the tensor
# because mem location is the same

#PyTorch domain libraries provide a number of sample preloaded datasets
# (such as FashionMNIST) that subclass torch.utils.data.Dataset and
# implement functions specific to the particular data.
# root is the path where the train/test data is stored.
# train specifies training or test dataset.
# download=True downloads the data from the Internet if it's
# not available at root.
# transform and target_transform specify the feature and label
# transformations.

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
label_name = list(labels_map.values())[label]
print(f"Label: {label_name}")

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


print("neural network attempt \n")