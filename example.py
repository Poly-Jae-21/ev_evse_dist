import torch

# Create a tensor
x = torch.randn(2, 3)

# Move it to CUDA if available
if torch.cuda.is_available():
    x = x.to('cuda')

# Check if the tensor is on CUDA
print("Is tensor on CUDA?", x.is_cuda)

# For a PyTorch model:
# model = YourModel()
# if torch.cuda.is_available():
#     model.to('cuda')
# print("Is model on CUDA?", next(model.parameters()).is_cuda)