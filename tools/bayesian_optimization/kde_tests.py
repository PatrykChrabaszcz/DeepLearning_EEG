from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np

# Check what happens when we have categorical 1, 2 and we request probability of categorical 0
data = np.array([1,1,1,1,1,1,1,1,1,1,1]).reshape([11,1])


def bw(data):
    X = np.std(data, axis=0)
    nobs = data.shape[0]

    return 1.06 * X * nobs ** (- 1. / (4 + data.shape[1]))


kde_e = KDEMultivariate(data, bw=np.array([0.5]), var_type='u')

print(bw(data))
print(kde_e.bw)

print(kde_e.pdf([[2.]]))
# test_point = np.array([[0.]])
#
# print(kde_e.data)
# print(kde_e.pdf(test_point))
