import numpy as np

from two_layer_perceptron import TwoLayerPerceptron
from keras.datasets import mnist

np.set_printoptions(formatter={"float": lambda x: "{0:0.1f}".format(x)})

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape([X_train.shape[0], 784])
X_test = X_test.reshape([X_test.shape[0], 784])

model = TwoLayerPerceptron(
    cost_func="cross-entropy",
    hidden_activ_func="ReLU",
    hidden_neurons_number=49,
    learning_rate=5e-6,
    convergence_change=5e-2,
    epochs_no_change_limit=10,
    max_epochs_number=200,
)

model.fit(X_train, y_train, debug=True)
print(f"Final score on the test set: {model.score(X_test,y_test)}%")
