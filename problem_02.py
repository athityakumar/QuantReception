# def solution(cls, a, theta, X0, sigma):
#     '''
#     input1 : double
#     input2 : double
#     input3 : double
#     input4 : double
    
#     Expected return type : List[Double]
#     '''
    # Read only region end
    # Write code here
from math import sqrt
from numpy.random import randn, randint

a = 0.8
theta = 100
X0 = 100
sigma = 0.2
n = 100
T = 2.0
m = 10000

def compute_multiverse(a, theta, sigma, X0):
    times = [0] + [T*t/100 for t in range(1, n+1)]
    # x_multiverse = randfloat(1, size=(m, n+1))
    x_multiverse = []
    delta_t = T/n
    for j in range(m):
        x_universe = [X0]
        randomness = randn(n)
        for (i, t) in enumerate(times[:-1]):
            delta_w = randomness[i] * sqrt(delta_t)
            delta_x = a*(theta - x_universe[-1])*delta_t + sigma * sqrt(x_universe[-1]) * delta_w
            x_universe.append(x_universe[-1] + delta_x)
        x_multiverse.append(x_universe)
    return x_multiverse

def fetch_xT(x_multiverse):
    return [x[-1] for x in x_multiverse]

def fetch_max_xT_wrt_100(x_multiverse):
    xT = fetch_xT(x_multiverse)
    max_xT_wrt_100 = [x-100 if x>100 else 0 for x in xT]
    return max_xT_wrt_100

def mean(elements):
    return sum(elements)/len(elements)

x_multiverse = compute_multiverse(a, theta, sigma, X0)
print x_multiverse[0]
print x_multiverse[-1]
xT = fetch_xT(x_multiverse)
# print xT
max_xT_wrt_100 = fetch_max_xT_wrt_100(x_multiverse)
# print max_xT_wrt_100
mean_of_xT = mean(xT)
mean_of_xT_wrt_100 = mean(max_xT_wrt_100)

sigmas = [(i+1)/20.0 for i in range(20)]
mean_of_xT_wrt_100_new = []

for new_sigma in sigmas:
    x_multiverse_new = compute_multiverse(a, theta, sigma, X0)
    max_xT_wrt_100_new = fetch_max_xT_wrt_100(x_multiverse_new)
    mean_of_xT_wrt_100_new.append(mean(max_xT_wrt_100_new))

derivative = []
for i in range(19):
    derivative.append((mean_of_xT_wrt_100_new[i+1]-mean_of_xT_wrt_100_new[i])/(sigmas[i+1]-sigmas[i]))

expectation_of_derivative = max(derivative)
print [mean_of_xT, mean_of_xT_wrt_100, expectation_of_derivative]
