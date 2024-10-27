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
        # Save predictions and true labels for backward pass
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = self.y_pred.shape[0]
        """ for i in range(batch_size):
            print(np.sum(self.y_pred[i])) """
        # Calculate the cross-entropy loss for each sample
        loss = - np.log(self.y_pred[np.arange(batch_size), self.y_true])
        """ for i in range(batch_size):
            print(loss[i]) """
        
        return loss

    def backward(self, grad: np.ndarray) -> np.ndarray:
        
        batch_size = self.y_pred.shape[0]
        
        # Calculate the gradient with respect to the predicted probabilities
        grad_output = np.zeros_like(self.y_pred)
        grad_output[np.arange(batch_size), self.y_true] = -1 / self.y_pred[np.arange(batch_size), self.y_true]
        
        # Normalize the gradient by batch size and multiply by the incoming gradient
        result = grad_output * grad[:, None]
        """ for i in range(batch_size):
            print(np.sum(result[i])) """

        # 使用 numpy.nditer 迭代每個元素並格式化輸出
        """ it = np.nditer(result, flags=['multi_index'])
        while not it.finished:
            print(f"{it[0]:.5f}", end=' ')
            if it.multi_index[1] == result.shape[1] - 1:
                print()  # 換行
            it.iternext() """

        return result
