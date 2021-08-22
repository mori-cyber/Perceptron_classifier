import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# _______________________________________________________________________________
# perceptron class


class perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(x):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


    def evaluate(self,y_test, y_pred):
        MAE = np.absolute(np.subtract(y_test, y_pred)).mean()
        return MAE


# -------------------------------------------------------------------------------------------------
train_data = pd.read_csv('linear_data_train.csv')
# print(train_data.head())
X_train = train_data[train_data.columns[0:2]]
Y_train = train_data[train_data.columns[-1]]
X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(-1, 1)
test_data = pd.read_csv('linear_data_test.csv')

X_test = test_data[test_data.columns[0:2]]
Y_test = test_data[test_data.columns[-1]]
X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(-1, 1)
# ----------------------------------------------------------------------------------------------
model = perceptron()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
dim = model.weights

print('loss of model is:', model.evaluate(Y_test, predictions))
# --------------------------------------------------------------------------------------------------
# classification plot
# ----------------------------------------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#
ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, color='0',
           marker='o', s=4, edgecolor='0')

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 1])
x1_1 = (-model.weights[0] * x0_1 - model.bias) / model.weights[1]
x1_2 = (-model.weights[0] * x0_2 - model.bias) / model.weights[1]
x, y = np.meshgrid([x0_1, x0_2], [x1_1, x1_2])
z = x * dim[0] + y * dim[1]
ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.4)

ax.set_title('linear_data')
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")

plt.show()
# -----------------------------------------------------------------------------------------------
