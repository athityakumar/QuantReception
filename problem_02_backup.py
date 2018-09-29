  from math import sqrt
   from numpy.random import randn

    n = 100
     T = 2.0
      m = 10000

       def compute_multiverse(a, theta, sigma, X0):
            times = [0] + [T*t/100 for t in range(1, n+1)]
            x_multiverse = []
            delta_t = T/n
            for j in range(m):
                x_universe = [X0]
                randomness = randn(n)
                for (i, t) in enumerate(times[:-1]):
                    delta_w = randomness[i] * sqrt(delta_t)
                    delta_x = a * \
                        (theta - x_universe[-1])*delta_t + \
                        sigma * sqrt(x_universe[-1]) * delta_w
                    x_universe.append(x_universe[-1] + delta_x)
                x_multiverse.append(x_universe)
            return x_multiverse

        def fetch_xT(x_multiverse):
            return [x[-1] for x in x_multiverse]

        def fetch_max_xT_wrt_100(x_multiverse):
            xT = fetch_xT(x_multiverse)
            max_xT_wrt_100 = [x-100 if x > 100 else 0 for x in xT]
            return max_xT_wrt_100

        def mean(elements):
            return sum(elements)/len(elements)

        x_multiverse = compute_multiverse(a, theta, sigma, X0)
        xT = fetch_xT(x_multiverse)
        max_xT_wrt_100 = fetch_max_xT_wrt_100(x_multiverse)
        mean_of_xT = mean(xT)
        mean_of_xT_wrt_100 = mean(max_xT_wrt_100)

        delta_sigma = 0.1
        sigma_new = sigma + delta_sigma

        x_multiverse_new = compute_multiverse(a, theta, sigma_new, X0)
        xT_new = fetch_xT(x_multiverse_new)
        max_xT_wrt_100_new = fetch_max_xT_wrt_100(x_multiverse_new)
        mean_of_xT_wrt_100_new = mean(max_xT_wrt_100_new)

        expectation_of_derivative = (
            mean_of_xT_wrt_100_new - mean_of_xT_wrt_100) / delta_sigma

        return [mean_of_xT, mean_of_xT_wrt_100, expectation_of_derivative]
