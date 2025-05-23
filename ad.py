import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

loss.backward()
print(w.grad)
print(b.grad)


# By default, all tensors with requires_grad=True are tracking their
# computational history and support gradient (computation.
# However, there) are some cases when we don't need to do that,
# for example, when we've trained the model and just want to apply
# it  to some input data; that is, we only want to do forward
# computations through the network. We can stop tracking
# computations  by surrounding our computation code with a
# torch.no_grad() block:

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)