# License: BSD 3 clause

# from .base import ModelHawkes, ModelSecondOrder, ModelSelfConcordant, \
#    LOSS_AND_GRAD
from tick.optim.model.build.model import ModelHawkesCustom as \
    ModelHawkesCustom

import numpy as np
from scipy.optimize import check_grad
import matplotlib as plt

beta = 2.0
MaxN_of_f = 5

timestamps = [np.array([0.31, 0.93, 1.29, 2.32, 4.25]),
              np.array([0.12, 1.19, 2.12, 2.41, 3.35, 4.21])]
T = 4.5

coeffs = np.array([1., 3., 2., 3., 4., 1, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10])
# corresponding to mu, alpha,f_i(n)

TestObj = ModelHawkesCustom(beta, MaxN_of_f)
TestObj.set_data(timestamps, T)

''' test of the loss function'''
loss_out = TestObj.loss(coeffs)
print(loss_out)

'''test of the grad'''
grad_out = np.array(np.zeros(len(coeffs)))
TestObj.grad(coeffs, grad_out)
print(grad_out)


def custom_loss(coeffs, *argv):
    self = argv[0]
    return self.loss(coeffs)


def custom_grad(coeffs, *argv):
    self = argv[0]
    grad_out = np.array(np.zeros(len(coeffs)))
    self.grad(coeffs, grad_out)
    return grad_out


print(custom_loss(coeffs, TestObj))
print(custom_grad(coeffs, TestObj))

print('#' * 40, '\nTest of gradient\n', '#' * 40)
# print(check_grad(custom_loss, custom_grad, coeffs, TestObj))

manual_grad = []
dup_coeffs = coeffs.copy()
epsilon = 1e-8
for i in range(len(coeffs)):
    dup_coeffs[i] += epsilon
    loss_out_new = custom_loss(dup_coeffs, TestObj)
    manual_grad.append((loss_out_new - loss_out) / epsilon)
    dup_coeffs[i] -= epsilon
manual_grad = np.array(manual_grad)
print(manual_grad)


# class ModelHawkesCustom(ModelHawkes,
#                                     ModelSecondOrder,
#                                     ModelSelfConcordant):
#     # In Hawkes case, getting value and grad at the same time need only
#     # one pas over the data
#     pass_per_operation = \
#         {k: v for d in [ModelSecondOrder.pass_per_operation,
#                         {LOSS_AND_GRAD: 1}] for k, v in d.items()}
#
#     _attrinfos = {
#         "decay": {
#             "cpp_setter": "set_decay"
#         },
#     }
#
#     __init__(self, decay: 'double const', MaxN_of_f: 'unsigned long const', n_cores: 'unsigned int const' = 1):
#
#     def __init__(self, decay: float, MaxN_of_f : int, n_threads: int = 1):
#         ModelHawkes.__init__(self, n_threads=1, approx=0)
#         ModelSecondOrder.__init__(self)
#         ModelSelfConcordant.__init__(self)
#         self.decay = decay
#         self._MaxN_of_f = MaxN_of_f
#         self._model = _ModelHawkesCustom(decay, MaxN_of_f, n_threads)
#
#     def fit(self, events, end_times=None):
#         """Set the corresponding realization(s) of the process.
#
#         Parameters
#         ----------
#         events : `list` of `list` of `np.ndarray`
#             List of Hawkes processes realizations.
#             Each realization of the Hawkes process is a list of n_node for
#             each component of the Hawkes. Namely `events[i][j]` contains a
#             one-dimensional `numpy.array` of the events' timestamps of
#             component j of realization i.
#             If only one realization is given, it will be wrapped into a list
#
#         end_times : `np.ndarray` or `float`, default = None
#             List of end time of all hawkes processes that will be given to the
#             model. If None, it will be set to each realization's latest time.
#             If only one realization is provided, then a float can be given.
#         """
#         ModelSecondOrder.fit(self, events)
#         ModelSelfConcordant.fit(self, events)
#         return ModelHawkes.fit(self, events, end_times=end_times)
#
#     @property
#     def decays(self):
#         return self.decay
#
#     @property
#     def _epoch_size(self):
#         # This gives the typical size of an epoch when using a
#         # stochastic optimization algorithm
#         return self.n_jumps
#
#     @property
#     def _rand_max(self):
#         # This allows to obtain the range of the random sampling when
#         # using a stochastic optimization algorithm
#         return self.n_jumps
