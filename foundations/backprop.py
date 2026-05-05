import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        z = np.dot(x, w) + b
        y_hat = sigmoid(z)

        L = 0.5*((y_true - y_hat)**2)

        dl_dw = -1 * (y_true-y_hat) * (sigmoid(z)*(1-sigmoid(z))) * x
        dl_db = -1 * (y_true-y_hat) * (sigmoid(z)*(1-sigmoid(z))) * 1

        return (np.round(dl_dw, 5), np.round(dl_db, 5))
