from Network import*
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from Data import*

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# keep 0 - 8
train_filter = np.where((y_train == 0) | (y_train == 8))
test_filter = np.where((y_test == 0) | (y_test == 8))
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

# reshape and normalize
x_train = x_train.reshape(-1, 28*28, 1)
x_test = x_train.reshape(-1, 28*28, 1)

x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255

y_train = (y_train==8)*1
y_test = (y_test==8)*1

# split in 2 categories and make equal
x0_train = x_train[np.nonzero((y_train == 0))]
x1_train = x_train[np.nonzero((y_train == 1))]

x0_test = x_test[np.nonzero((y_test == 0))]
x1_test = x_test[np.nonzero((y_test == 1))]

y0_train = y_train[np.nonzero((y_train == 0))]
y1_train = y_train[np.nonzero((y_train == 1))]

y0_test = y_test[np.nonzero((y_test == 0))]
y1_test = y_test[np.nonzero((y_test == 1))]

x0_train = x0_train[:5851]
y0_train = y0_train[:5851]

x0_test = x0_test[:974]
y0_test = y0_test[:974]

x_train = [x0_train, x1_train]
y_train = [y0_train, y1_train]


# hinge network
nn_hi = Network()
nn_hi.add(FCLayer(784, 300))
nn_hi.add(ActivationLayer(relu, d_relu))
nn_hi.add(FCLayer(300, 1))
# train
nn_hi.use(hinge, d_hinge)
err_hi, ep_hi = nn_hi.fit(x_train, y_train, epochs=40,learning_rate=0.000001)
#evaluate
r0 = nn_hi.predict(x0_test)
r1 = nn_hi.predict(x1_test)
# likelihood ratio test
H0_error = H_error_test(r0, 0)[1]
H1_error = H_error_test(r1, 0)[0]
tot_error = (H0_error + H1_error) / (2*len(x0_test)) * 100



# cross entropy
nn_ce = Network()
nn_ce.add(FCLayer(784, 300))
nn_ce.add(ActivationLayer(relu, d_relu))
nn_ce.add(FCLayer(300, 1))
nn_ce.add(ActivationLayer(sigmoid, d_sigmoid))
# train
nn_ce.use(cross_entropy, d_cross_entropy)
err_ce, ep_ce = nn_ce.fit(x_train, y_train, epochs=40,learning_rate=0.000001)
#evaluate
r0 = nn_ce.predict(x0_test)
r1 = nn_ce.predict(x1_test)
# likelihood ratio test
H0_error = H_error_test(r0, 0.5)[1]
H1_error = H_error_test(r1, 0.5)[0]
tot_error = (H0_error + H1_error) / (len(x0_test) + len(x1_test)) * 100


# Exponential
nn_ex = Network()
nn_ex.add(FCLayer(784, 300))
nn_ex.add(ActivationLayer(relu, d_relu))
nn_ex.add(FCLayer(300, 1))
# train
nn_ex.use(exponential, d_exponential)
err_ex, ep_ex = nn_ex.fit(x_train, y_train, epochs=40, learning_rate=0.000001)
#evaluate
r0 = nn_ex.predict(x0_test)
r1 = nn_ex.predict(x1_test)
# likelihood ratio test
H0_error = H_error_test(r0, 0)[1]
H1_error = H_error_test(r1, 0)[0]
tot_error = (H0_error + H1_error) / (2*len(x0_test)) * 100



