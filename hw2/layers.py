import abc
from typing import List

import numpy as np


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propogate through the layer.

        Parameters:
            x (`np.ndarray`):
                input tensor with shape `(batch_size, input_dim)`.

        Returns:
            out (`np.ndarray`):
                output tensor of the layer with shape
                `(batch_size, output_dim)`.
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward propogate through the layer.

        Calculate the gradient of the loss with respect to both the input and
        the parameters of the layer. The gradients with respect to the
        parameters are summed over the batch.

        Parameters:
            grad (`np.ndarray`):
                gradient with respect to the output of the layer.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of the layer.
        """
        return NotImplemented
        
    def params(self) -> List[np.ndarray]:
        """Return the parameters of the parameters.

        The order of parameters should be the same as the order of gradients
        returned by `self.grads()`.
        """
        return []

    def grads(self) -> List[np.ndarray]:
        """Return the gradients of the parameters.

        The order of gradients should be the same as the order of parameters
        returned by `self.params()`.
        """
        return []


class Sequential(Layer):
    """A sequential model that stacks layers in a sequential manner.

    Parameters:
        layers (`List[Layer]`):
            list of layers to be added to the model.
    """
    def __init__(self, *layers: List[Layer]):
        self.layers = []
        self.append(*layers)

    def append(self, *layers: List[Layer]):
        """Add layers to the model"""
        self.layers.extend(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Sequentially forward propogation through layers.

        Parameters:
            x (`np.ndarray`):
                input tensor.

        Returns:
            out (`np.ndarray`):
                output tensor of the last layer.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Sequentially backward propogation through layers.

        Parameters:
            grad (`np.ndarray`):
                gradient of loss function with respect to the output of forward
                propogation.

        Returns:
            grad (`np.ndarray`):
                gradient with respect to the input of forward propogation.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self) -> List[np.ndarray]:
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.params())
        return parameters

    def grads(self) -> List[np.ndarray]:
        gradients = []
        for layer in self.layers:
            gradients.extend(layer.grads())
        return gradients


class MatMul(Layer):
    """Matrix multiplication layer.
    
    Parameters:
        W (`np.ndarray`):
            weight matrix with shape `(input_dim, output_dim)`
    """
    def __init__(self, W: np.ndarray):
        #assert W.ndim == 2, "Weight matrix W must be 2D"
        self.W = W  # Weight matrix
        self.x = None  # To store input for the backward pass
        self.grad_W = None  # To store the gradient with respect to W

    def forward(self, x: np.ndarray) -> np.ndarray:
        #assert x.ndim == 2, "Input x must be 2D"
        self.x = x  # Save input for the backward pass
        return x @ self.W  # Perform matrix multiplication

    def backward(self, grad: np.ndarray) -> np.ndarray:
        #assert grad.ndim == 2, "Gradient grad must be 2D"
        grad_input = grad @ self.W.T  # Gradient with respect to input x
        self.grad_W = self.x.T @ grad  # Gradient with respect to weights W
        return grad_input

    def params(self) -> List[np.ndarray]:
        return [self.W]

    def grads(self) -> List[np.ndarray]:
        return [self.grad_W]


class Bias(Layer):
    """Bias layer.

    Parameters:
        b (`np.ndarray`):
            bias vector with shape `(output_dim,)`.
    """
    def __init__(self, b: np.ndarray):
        self.b = b # Bias vector
        self.grad_b = None  # To store the gradient with respect to the bias
        self.x = None  # To store input for the backward pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x  # Save input for the backward pass
        return x + self.b  # Add the bias to each input element-wise

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # Compute the gradient with respect to the bias
        self.grad_b = np.sum(grad, axis=0)  # Sum over batch to get gradient with respect to bias
        return grad  # The gradient with respect to input is unchanged

    def params(self) -> List[np.ndarray]:
        return [self.b]

    def grads(self) -> List[np.ndarray]:
        return [self.grad_b]


class ReLU(Layer):
    """Rectified Linear Unit."""

    def forward(self, x):
        self.cache = x
        return np.clip(x, 0, None)

    def backward(self, grad):
        x = self.cache
        return np.where(x > 0, grad, 0)


class Softmax(Layer):
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # For numerical stability, we shift the input by subtracting max
        self.cache = x
        exp_x = np.exp(x)
        sum_exp_x = exp_x.sum(axis=1, keepdims=True)
        self.y = exp_x / sum_exp_x
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        y = self.forward(self.cache)
        return grad - y*(grad.sum(axis = 0))  # Shape: (batch_size, num_classes)
