import numpy as np
import abc


class Loss(abc.ABC):
    @abc.abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute the loss between the predicted and true values.

        Parameters:
            y_pred (`np.ndarray`):
                The predicted values, typically output from a model. The shape
                of `y_pred` is (`batch_size`, `input_dim`).
            y_true (`np.ndarray`):
                The true/target values corresponding to the predictions. Note
                that the shapes of `y_pred` and `y_true` might not match, but
                they will have the same number of samples, i.e. `batch_size`.

        Returns:
            loss (`np.ndarray`):
                The loss value for each sample in the input. The shape of the
                output is (`batch_size`,).
        """
        return NotImplemented

    @abc.abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss w.r.t. the predicted values.

        Parameters:
            grad (`np.ndarray`):
                The gradient of the final loss with respect to the output of
                `self.forward`. This value typically reflects a scaling factor
                caused by averaging the loss, and can also be used to reverse
                the gradient direction to achieve gradient ascent.

        Returns:
            grad (`np.ndarray`):
                The gradient of the loss with respect to the `y_pred`.
        """
        return NotImplemented


class CrossEntropy(Loss):
    def __init__(self):
        self.y_pred = None  # Store the predicted probabilities
        self.y_true = None  # Store the true labels
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute the cross-entropy loss between the predicted probabilities and
        the true labels.

        Parameters:
            y_pred (`np.ndarray`):
                A 2D array of predicted probabilities, where the first axis
                corresponds to the number of samples and the second axis
                corresponds to the number of classes.
            y_true (`np.ndarray`):
                A 1D array of true labels, where each label is an integer in
                the range [0, num_classes).
        """
        # Convert labels to one-hot encoding
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = self.y_pred.shape[0]

        # Calculate the cross-entropy loss
        loss = -np.emath.log(self.y_pred[np.arange(batch_size), self.y_true])
        return loss

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size = self.y_pred.shape[0]
        # Convert labels to one-hot encoding
        y_one_hot = np.zeros_like(self.y_pred)
        #print(y_one_hot.shape)
        #print(grad.shape)
        y_one_hot[np.arange(batch_size), self.y_true] = -1 / self.y_pred[np.arange(batch_size), self.y_true]
        #y_one_hot = y_one_hot / batch_size
        y_out = grad.T @ y_one_hot
        #print(y_out.shape)
        # Calculate the gradient
        return y_out
