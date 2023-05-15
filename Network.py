import numpy as np

class Network:
    def __init__(self):
            self.layers = []
            self.loss = None
            self.d_loss = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, d_loss):
        self.loss = loss
        self.d_loss = d_loss

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return np.array(result)

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train[0])
        err_arr = []
        for _ in range(epochs):
            err0 = [0]
            err1 = [0]
            for i in range(samples):
                # output0
                output0 = np.copy(x_train[0][i])
                for layer in self.layers:
                    output0 = layer.forward_propagation(output0)
                # output1
                output1 = np.copy(x_train[1][i])
                for layer in self.layers:
                    output1 = layer.forward_propagation(output1)
                err0.append(err0[i-1] + self.loss(y_train[0][i], output0))
                err1.append(err0[i-1] + self.loss(y_train[1][i], output1))
                error = self.d_loss(y_train[0][i], output0) +  self.d_loss(y_train[1][i],output1)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            #εξομαλυνση με 20 τελευταιες τιμες
            #err = sum(err0[-20:])/len(err0) + sum(err1[-20:])/len(err1)
            err = sum(err0)/ len(err0) + sum(err1) / len(err1)
            err_arr.append(err[0][0])
            #print('epoch %d/%d error=%f' % (_+1, epochs, err))
        return err_arr, epochs










class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.normal(0, 1 / (input_size + output_size), (output_size, input_size))
        self.offset = np.zeros((output_size, 1))
        self.l = 1
        self.E_weights = 0
        self.E_offset = 0

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.offset
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)
        # Normalize Derivatives
        self.E_weights = self.E_weights * (1 - self.l) + self.l * np.square(weights_error)
        self.E_offset = self.E_offset * (1 - self.l) + self.l * np.square(output_error)
        self.l = 0.005
        # Gradient descent
        self.weights -= learning_rate * weights_error / np.sqrt(self.E_weights + 10 ** -6)
        self.offset -= learning_rate * output_error / np.sqrt(self.E_offset + 10 ** -6)
        return input_error



class ActivationLayer:
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    # Returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.d_activation(self.input) * output_error






# ACTIVATION FUNCTIONS
# Relu
def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return np.exp(-x) / np.square(1 + np.exp(-x))



# LOSS FUNCTIONS
# Cross entropy
def cross_entropy(y_true, y_pred):
    if y_true == 0:
        return -np.log(1 - y_pred)
    elif y_true == 1:
        return -np.log(y_pred)
def d_cross_entropy(y_true, y_pred):
    if y_true == 0:
        return 1 / (1 - y_pred)
    elif y_true == 1:
        return -1 / y_pred

# Exponential
def exponential(y_true, y_pred):
    if y_true == 0:
        return np.exp(0.5 * y_pred)
    elif y_true == 1:
        return np.exp(-0.5 * y_pred)
def d_exponential(y_true, y_pred):
    if y_true == 0:
        return 0.5 * np.exp(0.5 * y_pred)
    elif y_true == 1:
        return -0.5 * np.exp(-0.5 * y_pred)

# Hinge
def hinge(y_train, y_pred):
    if y_train == 0:
        return np.maximum(0, 1 + y_pred)
    elif y_train == 1:
        return np.maximum(0, 1 - y_pred)
def d_hinge(y_train, y_pred):
    y_pred[y_pred<=0] = 0
    if y_train == 0:
        y_pred[y_pred>0] = 1
        return y_pred
    elif y_train == 1:
        y_pred[y_pred>0] = -1
        return y_pred
