import numpy as np

class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = np.array(data)
    self.grad = None
    self.ctx = None
    self.requries_grad = requires_grad

  #def __repr__(self):

  def __neg__(self): raise NotImplementedError

  def __add__(self, x): raise NotImplementedError
  def __sub__(self, x): raise NotImplementedError
  def __mul__(self, x): raise NotImplementedError
  def __pow__(self, x): raise NotImplementedError
  def __truediv__(self, x): raise NotImplementedError
  def __matmul__(self, x): raise NotImplementedError

  def __radd__(self, x): raise NotImplementedError
  def __rsub__(self, x): raise NotImplementedError
  def __rmul__(self, x): raise NotImplementedError
  def __rpow__(self, x): raise NotImplementedError
  def __rtruediv__(self, x): raise NotImplementedError
  def __rmatmul__(self, x): raise NotImplementedError

  def __iadd__(self, x): raise NotImplementedError
  def __isub__(self, x): raise NotImplementedError
  def __imul__(self, x): raise NotImplementedError
  def __ipow__(self, x): raise NotImplementedError
  def __itruediv__(self, x): raise NotImplementedError
  def __imatmul__(self, x): raise NotImplementedError

class Function:
  def apply(self, *args):
    pass

  def forward(self, *args): raise NotImplementedError
  def backward(self, *args): raise NotImplementedError