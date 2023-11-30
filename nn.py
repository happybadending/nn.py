import numpy as np

class Tensor:
  def __init__(self, data, requires_grad=False):
    self.data = np.array(data)
    self.grad = None
    self.ctx = None
    self.requries_grad = requires_grad

  def __repr__(self):
    return f"{self.data}"

  def __neg__(self): return Neg.apply(self)

  def __add__(self, x): return Add.apply(self, x)
  def __sub__(self, x): return Sub.apply(self, x)
  def __mul__(self, x): return Mul.apply(self, x)
  def __pow__(self, x): return Pow.apply(self, x)
  def __truediv__(self, x): return Div.apply(self, x)
  def __matmul__(self, x): return MatMul.apply(self, x)

  def backward(self):
    def toposort(node, visited, ret):
      return ret
    self.grad = Tensor(1)
    for node in reversed(toposort(self, set(), [])):
      pass

class Function:
  @classmethod
  def apply(fn, *x): return Tensor(fn().forward(*[t.data for t in x]))

  def forward(self, *args): raise NotImplementedError
  def backward(self, *args): raise NotImplementedError

class Neg(Function):
  def forward(self, x): return -x

class Add(Function):
  def forward(self, x, y): return x + y
  def backward(self, grad): return

class Sub(Function):
  def forward(self, x, y): return x - y

class Mul(Function):
  def forward(self, x, y): return x * y
  def backward(self, grad): return

class Pow(Function):
  def forward(self, x, y): return x ** y

class Div(Function):
  def forward(self, x, y): return x / y

class MatMul(Function):
  def forward(self, x, y): return x @ y