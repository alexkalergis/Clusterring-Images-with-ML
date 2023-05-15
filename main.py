from Network import*
from Data import*
import numpy as np
import matplotlib.pyplot as plt

#----------Create the 10^6 Data for the BAYES test----------
N_test = 1000000
(x1,x2) = create_data(N_test)
x1_test = x1
x2_test = x2
# likelihood ratio
r0 = f1(x1_test[:,0]) * f1(x1_test[:,1]) / (f0(x1_test[:,0]) * f0(x1_test[:,1]))
r1 = f1(x2_test[:,0]) * f1(x2_test[:,1]) / (f0(x2_test[:,0]) * f0(x2_test[:,1]))
# H_error_test
H0_error = H_error_test(r0, 1)[1]
H1_error = H_error_test(r1, 1)[0]
# Bayes test
tot_error = 0.5*(H0_error/N_test + H1_error/N_test )
# Print the Total error
#print("ERROR from f0(x):", (H0_error/N_test)*100, "%")
#print("ERROR from f1(x):", (H1_error/N_test)*100, "%")
#print("BAYES TOTAL ERROR:", tot_error*100, "%")

#----------Network----------
# Create train data
N_train = 200
x1_train, x2_train = create_data(N_train)
# Train data
x1_train = x1_train.reshape((N_train, 2, 1))
x2_train = x2_train.reshape((N_train, 2, 1))
y1_train = np.zeros((N_train, 1))
y2_train = np.ones((N_train, 1))
x_train = [x1_train, x2_train]
y_train = [y1_train, y2_train]
# Test data
x1_test = x1_test.reshape(N_test, 2, 1)
x2_test = x2_test.reshape(N_test, 2, 1)

#----------Cross-entropy network----------
nn_ce = Network()
nn_ce.add(FCLayer(2, 20))
nn_ce.add(ActivationLayer(relu, d_relu))
nn_ce.add(FCLayer(20, 1))
nn_ce.add(ActivationLayer(sigmoid, d_sigmoid))
# Train
nn_ce.use(cross_entropy, d_cross_entropy)
err_ce, ep_ce = nn_ce.fit(x_train, y_train, epochs=1000, learning_rate=0.001)
# likelihood ratio test prediction
r0 = nn_ce.predict(x1_test)
r1 = nn_ce.predict(x2_test)
# H_error_test
H0_error = H_error_test(r0, 0.5)[1]
H1_error = H_error_test(r1, 0.5)[0]
tot_error = 0.5*(H0_error/N_test + H1_error/N_test)
# Print the Total error

print("ERROR from f0(x):", (H0_error/N_test)*100, "%")
print("ERROR from f1(x):", (H1_error/N_test)*100, "%")
print("CROSS_ENTROPY TOTAL ERROR:", tot_error*100, "%")
plt.plot(err_ce, color='b', label='cross-entropy')
plt.xlabel('epochs')
plt.ylabel('error')
plt.title('Cross-entropy Error Function')
plt.legend()
plt.show()


#----------Exponential network----------
nn_ex = Network()
nn_ex.add(FCLayer(2, 20))
nn_ex.add(ActivationLayer(relu, d_relu))
nn_ex.add(FCLayer(20, 1))
# Train
nn_ex.use(exponential, d_exponential)
err_ex, ep_ex = nn_ex.fit(x_train, y_train, epochs=1500,learning_rate=0.001)
# Likelihood ratio test prediction
r0 = nn_ex.predict(x1_test)
r1 = nn_ex.predict(x2_test)
# H_error_test
H0_error = H_error_test(r0, 0)[1]
H1_error = H_error_test(r1, 0)[0]
tot_error = 0.5*(H0_error/N_test + H1_error/N_test)
# Print the Total error
print("ERROR from f0(x):", (H0_error/N_test)*100, "%")
print("ERROR from f1(x):", (H1_error/N_test)*100, "%")
print("EXPONENTIAL TOTAL ERROR:", tot_error*100, "%")
plt.plot(err_ex, color='b', label='exponential')
plt.xlabel('epochs')
plt.ylabel('error')
plt.title('Exponential Error Function')
plt.legend()
plt.show()