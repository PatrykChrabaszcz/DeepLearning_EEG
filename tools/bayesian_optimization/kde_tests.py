from statsmodels.nonparametric.kernel_density import KDEMultivariate
import numpy as np


# Check what happens when we have categorical 1, 2 and we request probability of categorical 0
data = np.array([[0.00000001], [0.000000001], [0.000000001], [0.000000001], [0.00000001], [0.], [0.], [0.]])
kde_e = KDEMultivariate(data, var_type='c')

test_point = np.array([[0.]])

print(kde_e.data)
print(kde_e.pdf(test_point))
