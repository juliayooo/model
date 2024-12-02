import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# look for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# 28 by 28 pixels in as input, total 784
# First layer takes the 784 and transforms them into 512. The next
# layers then transform those 512 to the next layer. The last layer
# outputs the output layer, 10, also the number of classes
#

# Creating an instance and running the class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# Softmax function predicts the input desities which prevents the
# initialzation of a ten dimensional garbage value tensor which
# would otherwise by created by calling model.forward().
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


print(f"First Linear weights: {model.linear_relu_stack[0].weight} \n")

print(f"First Linear biases: {model.linear_relu_stack[0].bias} \n")

input_image = torch.rand(3,28,28)
print(input_image.size())


# Flattening the image to a one dimesional array of 784 pixel
# values based on color value
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())


# applies a linear transformation on the input using its stored weights
# and biases. The grayscale value of each pixel in the input layer
# is connected to neurons in the hidden layer for calculation.
# The calculation the transformation uses is weight∗input+bias.
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# Relu introduces non linearity which trains the model past simply
# linear patterns


print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)


# Speaks to the output layer. Each of the ten classes return 0 or 1
# for true or false. The class with the highest probablity is the
# output
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# When training neural networks, the most frequently used algorithm
# is back propagation. In this algorithm, parameters (model
# weights)  are adjusted according to the loss function gradient
# with respect to the given parameter. The loss function calculates
# the difference between the expected output and the actual output
# that a neural network produces. The goal is to get the result of
# the loss function as close to zero as possible. The algorithm
# traverses backwards through the neural network to adjust the
# weights and bias to retrain the model. That's why it's called
# back propagation. This back and forward process of retraining
# the model over time to reduce the loss to 0 is called the
# gradient descent.

