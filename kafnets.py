import torch
from torch.nn import Module, Parameter
from torch.nn.init import normal_


class KAF(Module):
    """Kernel Activation Function (KAF) with optimized memory usage."""

    def __init__(self, num_parameters, D=20, boundary=3.0):
        """
        :param num_parameters: Number of neurons in the layer.
        :param D: Size of the dictionary (reduced for optimization).
        :param boundary: Range of the activation function.
        """
        super(KAF, self).__init__()
        self.num_parameters = num_parameters
        self.D = D

        # Initialize the fixed dictionary
        dict_values = torch.linspace(-boundary, boundary, self.D)
        self.register_buffer('dict', dict_values)

        # Rule of thumb for gamma
        interval = self.dict[1] - self.dict[0]
        sigma = 2 * interval  # Empirically chosen
        self.gamma = 0.5 / (sigma ** 2)

        # Mixing coefficients
        self.alpha = Parameter(torch.empty(1, self.num_parameters, self.D))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        normal_(self.alpha.data, std=0.3)

    def forward(self, input):
        """
        Forward pass of KAF.
        :param input: Tensor of shape (batch_size, num_parameters, *), where * indicates additional dimensions.
        :return: Tensor with the same shape as `input` after applying the kernel function.
        """
        self.dict = self.dict.to(input.device)  # Ensure dictionary is on the same device
        input_shape = input.shape

        # Flatten the input to apply KAF
        input_flat = input.view(-1, input.shape[-1])  # Shape: (batch_size * spatial_size, num_parameters)
        kernel_outputs = []

        # Chunked computation for memory efficiency
        chunk_size = 512  # Adjust this as needed based on GPU memory
        for i in range(0, input_flat.size(0), chunk_size):
            input_chunk = input_flat[i:i + chunk_size]
            kernel_chunk = torch.exp(-self.gamma * (input_chunk.unsqueeze(-1) - self.dict) ** 2)  # Shape: (chunk_size, num_parameters, D)
            chunk_output = torch.sum(kernel_chunk * self.alpha, dim=-1)  # Shape: (chunk_size, num_parameters)
            kernel_outputs.append(chunk_output)

        # Concatenate all processed chunks
        output_flat = torch.cat(kernel_outputs, dim=0)  # Shape: (batch_size * spatial_size, num_parameters)

        # Reshape back to the original input shape
        output = output_flat.view(*input_shape)
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(num_parameters={self.num_parameters}, D={self.D})"
