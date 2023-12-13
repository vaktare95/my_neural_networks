import numpy as np

# COST FUNCTIONS


def mse(Y, R):
    return np.sum((Y - R) ** 2)


def cross_entropy(Y, R) -> np.array:
    return np.sum(-R * np.log(Y))


# CALCULATING 1st LAYER GRADIENT


def grad_1_mse(N, W2, Y, R, H, Z1, jK, jL, hidden_activ_func: str, lambd=1e-12):
    G1 = np.zeros([len(jL), N])
    for n in range(N):
        yn = Y[:, [n]]
        rn = R[:, [n]]
        hn = H[:, [n]]
        z1n = Z1[:, [n]]

        fn = np.matmul((W2 - np.matmul(jK, np.matmul(yn.T, W2))).T, (yn - rn) * yn)

        if hidden_activ_func == "sigmoid":
            gn = (hn ** 2) * np.exp(-z1n) * fn
        elif hidden_activ_func == "sigmoid_ext":
            gn = ((hn + jL) ** 2) * np.exp(-z1n) * fn
        elif hidden_activ_func == "tanh":
            gn = (jL - hn ** 2) * fn
        elif hidden_activ_func == "ReLU":
            gn = ((z1n > 0) + jL) * fn
        G1[:, n] = gn[:, 0]

        return G1


def grad_1_ce(W2, Y, R, H, Z1, JLN, hidden_activ_func: str, lambd=1e-100):
    F = np.matmul(W2.T, (Y - R))

    if hidden_activ_func == "sigmoid":
        G1 = (H ** 2) * np.exp(-Z1) * F
    elif hidden_activ_func == "sigmoid_ext":
        G1 = ((H + JLN) ** 2) * np.exp(-Z1) * F
    elif hidden_activ_func == "tanh":
        G1 = (JLN - H ** 2) * F
    elif hidden_activ_func == "ReLU":
        G1 = ((Z1 > 0) + JLN) * F

    return G1


# CALCULATING 2nd LAYER GRADIENT


def grad_2_mse(Y, R, JKK):
    G2 = Y * (np.matmul(JKK, (Y * R)) + Y - R - np.matmul(JKK, (Y ** 2)))
    return G2


def grad_2_ce(Y, R):
    G2 = Y - R
    return G2


# ACTIVATION FUNCTIONS


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_ext(x):
    return 2 / (1 + np.exp(-x)) - 1


def reLU(x):
    negative_idxs = x < 0
    x[negative_idxs] = 0
    return x


# CONFIG DICTS

ALLOWED_COST_FUNCTIONS = {"MSE": mse, "cross-entropy": cross_entropy}
ALLOWED_HIDDEN_LAYER_ACTIVATION_FUNCTIONS = {
    "sigmoid": sigmoid,
    "sigmoid_ext": sigmoid_ext,
    "tanh": np.tanh,
    "ReLU": reLU,
}
