import numpy as np

#Create Data
def create_data(N):
    x1 = np.random.normal(0, 1, (N,2))
    x2 = np.random.normal(np.random.choice([-1, 1], (N,2)), np.ones((N,2)))
    return x1, x2

def normal_dist(mean, sd, x):
    return 1 / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square((x - mean)/sd))

def f0(x):
    return normal_dist(0, 1, x)

def f1(x):
    return 0.5 * (normal_dist(-1, 1, x) + normal_dist(1, 1, x))

def H_error_test(H, threshold):
    N = len(H)
    H0_count = np.count_nonzero(H < threshold)
    H1_count = N - H0_count
    return H0_count, H1_count
