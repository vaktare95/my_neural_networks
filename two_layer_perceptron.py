from logging import Logger
from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

import neural_network_algorithm_config as config

logger = Logger("neural_network_algorithm")


def softmax(x: np.ndarray) -> np.ndarray:
    x_exp_sum = sum(np.exp(x))
    return np.exp(x) / x_exp_sum


class NeuralNetworkAlgorithm:
    cost_func: str
    hidden_activ_func: str
    hidden_neurons_number: int
    classes_number: int

    max_epochs_number: int
    batch_size: int

    learning_rate: float
    convergence_change: float
    epochs_no_change_limit: int

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def __init__(
        self,
        cost_func: str,
        hidden_activ_func: str,
        hidden_neurons_number: int,
        classes_number: int = 10,
        max_epochs_number: int = 200,
        batch_size: int = 200,
        learning_rate: float = 1e-3,
        convergence_change: float = 1e-4,
        epochs_no_change_limit: int = 10,
    ):
        if cost_func in config.ALLOWED_COST_FUNCTIONS:
            self.cost_func = cost_func
        else:
            logger.error(
                f"Given cost function ({cost_func}) not allowed. Allowed values: \
                    {config.ALLOWED_COST_FUNCTIONS.keys()}"
            )
        if hidden_activ_func in config.ALLOWED_HIDDEN_LAYER_ACTIVATION_FUNCTIONS:
            self.hidden_activ_func = hidden_activ_func
        else:
            logger.error(
                f"Given cost function ({hidden_activ_func}) not allowed. Allowed values: \
                    {config.ALLOWED_HIDDEN_LAYER_ACTIVATION_FUNCTIONS.keys()}"
            )

        self.hidden_neurons_number = hidden_neurons_number
        self.classes_number = classes_number
        self.max_epochs_number = max_epochs_number
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.convergence_change = convergence_change
        self.epochs_no_change_limit = epochs_no_change_limit

    def check_nan_pixels(self, X: np.ndarray) -> np.ndarray:
        if np.isnan(X).any():
            logger.warning("N/A values in data. Inserting zeros instead.")
            np.nan_to_num(X)
        return X

    def delete_examples_without_class(self, X: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        indexes_to_delete = np.isnan(r)
        X = X[:, ~indexes_to_delete]
        r = np.expand_dims(r[~indexes_to_delete], axis=0)
        return X, r

    def normalize_train_data(self, X: np.ndarray):
        return 2 * X / np.max(X) - 1

    def create_binary_vector(self, vec):
        binary_vector = np.zeros(self.classes_number)
        number = vec[0]
        binary_vector[int(number)] = 1
        return binary_vector

    def transform_labels_matrix(self, R: np.ndarray) -> np.ndarray:
        inputs_number = R.shape[1]
        R_trans = np.concatenate([R, np.zeros([self.classes_number - 1, inputs_number])], axis=0)
        return np.apply_along_axis(self.create_binary_vector, 0, R_trans)

    def initialize_weights(self, P: int, L: int, K: int) -> None:
        W1_sd = np.sqrt(2 / (P + L))
        self.W1 = np.random.normal(0, scale=W1_sd, size=(L, P))
        self.b1 = np.zeros([L, 1])
        W2_sd = np.sqrt(2 / (L + K))
        self.W2 = np.random.normal(0, scale=W2_sd, size=(K, L))
        self.b2 = np.zeros([K, 1])

    def process_batch(
        self,
        X: np.ndarray,
        R: Optional[np.ndarray] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        N = X.shape[1]
        Z1 = np.matmul(self.W1, X) + np.tile(self.b1, N)
        H = config.ALLOWED_HIDDEN_LAYER_ACTIVATION_FUNCTIONS[self.hidden_activ_func](Z1)
        Z2 = np.matmul(self.W2, H) + np.tile(self.b2, N)
        Y = softmax(Z2)
        training_state = True if isinstance(R, np.ndarray) else False
        if training_state:
            c = config.ALLOWED_COST_FUNCTIONS[self.cost_func](Y, R)
            return Y, H, Z1, c
        else:
            return Y

    def update_wages(
        self,
        Y: np.ndarray,
        H: np.ndarray,
        Z1: np.ndarray,
        X: np.ndarray,
        R: np.ndarray,
        JKK: np.ndarray,
        JLN: np.ndarray,
        jK: np.ndarray,
        jL: np.ndarray,
        mi: float,
    ) -> None:
        N = X.shape[1]
        jN = np.ones([N, 1])

        if self.cost_func == "MSE":
            G2 = config.grad_2_mse(Y, R, JKK)
        elif self.cost_func == "cross-entropy":
            G2 = config.grad_2_ce(Y, R)

        self.W2 -= mi * np.matmul(G2, H.T)
        self.b2 -= mi * np.matmul(G2, jN)

        if self.cost_func == "MSE":
            G1 = config.grad_1_mse(N, self.W2, Y, R, H, Z1, jK, jL, self.hidden_activ_func)
        elif self.cost_func == "cross-entropy":
            G1 = config.grad_1_ce(self.W2, Y, R, H, Z1, JLN, self.hidden_activ_func)

        self.W1 -= mi * np.matmul(G1, X.T)
        self.b1 -= mi * np.matmul(G1, jN)

    def plot_convergence_progress(self, c_vec: np.ndarray):
        plt.plot(c_vec)

        plt.title("Convergence Progress")
        plt.xlabel("epoch number")
        plt.ylabel("cost function")

        plt.show()

    def fit(self, X: np.ndarray, r: np.ndarray, debug: bool = False, plot_cost: bool = False) -> None:
        X = self.check_nan_pixels(X.T)
        X, r = self.delete_examples_without_class(X, r)
        X = self.normalize_train_data(X)
        R = self.transform_labels_matrix(r)
        mi = self.learning_rate

        P = X.shape[0]
        L = self.hidden_neurons_number
        K = self.classes_number

        self.initialize_weights(P, L, K)

        inputs_number = X.shape[1]
        N = min(inputs_number, self.batch_size)
        batches_number = np.ceil(inputs_number / N)

        JKK = np.ones([K, K])
        JLN = np.ones([L, N])
        jK = np.ones([K, 1])
        jL = np.ones([L, 1])

        c_vec = np.zeros(self.max_epochs_number)
        c_prev = np.inf
        epochs_no_change = 0
        for epoch in range(self.max_epochs_number):
            shuffle_vec = np.random.randn(X.shape[1]).argsort(axis=0)

            X_shuffled = X[:, shuffle_vec]
            X_batches = np.array_split(X_shuffled, batches_number, axis=1)

            R_shuffled = R[:, shuffle_vec]
            R_batches = np.array_split(R_shuffled, batches_number, axis=1)

            for X_batch, R_batch in zip(X_batches, R_batches):
                Y, H, Z1, c = self.process_batch(X_batch, R_batch)
                self.update_wages(Y, H, Z1, X_batch, R_batch, JKK, JLN, jK, jL, mi)

            if debug:
                print(f"Epoch: {epoch}, cost_func: {c}")

            c_vec[epoch] = c

            if np.abs(c - c_prev) < self.convergence_change:
                epochs_no_change += 1
                if epochs_no_change == self.epochs_no_change_limit:
                    break
            else:
                epochs_no_change = 0
                c_prev = c

        if plot_cost:
            self.plot_convergence_progress(c_vec)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self.check_nan_pixels(X.T)
        X = self.normalize_train_data(X)
        Y = self.process_batch(X)
        return np.argmax(Y, axis=0)

    def score(self, X: np.ndarray, r: np.ndarray) -> float:
        y = self.predict(X)
        return 100 * np.sum(y == r) / len(y)
