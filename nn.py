import numpy as np

class Tensor:
  def __init__(self, data, ctx=None):
    self.data = np.array(data)
    self.grad = None
    self.ctx = ctx

  def __repr__(self): return f"<Tensor {self.data} grad {self.grad}>"
  def __getitem__(self, x): return Tensor(self.data[x])

  @property
  def shape(self): return self.data.shape
  @property
  def T(self): return Tensor(self.data.T)

  @staticmethod
  def tensor(x): return Tensor(x) if not isinstance(x, Tensor) else x
  def __neg__(self): return Neg.apply(self)
  def __add__(self, x): return Add.apply(self, self.tensor(x))
  def __sub__(self, x): return Sub.apply(self, self.tensor(x))
  def __mul__(self, x): return Mul.apply(self, self.tensor(x))
  def __pow__(self, x): return Pow.apply(self, self.tensor(x))
  def __truediv__(self, x): return Div.apply(self, self.tensor(x))
  def __matmul__(self, x): return MatMul.apply(self, self.tensor(x))

  def __radd__(self, x): return Add.apply(self.tensor(x), self)
  def __rsub__(self, x): return Sub.apply(self.tensor(x), self)
  def __rmul__(self, x): return Mul.apply(self.tensor(x), self)

  @staticmethod
  def argmax(*args, **kwargs): return Tensor(np.argmax(*args, **kwargs))
  @staticmethod
  def tanh(*args, **kwargs): return Tensor(np.tanh(*args, **kwargs))
  @staticmethod
  def tri(*args, **kwargs): return Tensor(np.tri(*args, **kwargs))
  @staticmethod
  def split(*args, **kwargs): return Tensor(np.split(*args, **kwargs))
  @staticmethod
  def sqrt(*args, **kwargs): return Tensor(np.sqrt(*args))
  @staticmethod
  def exp(*args, **kwargs): return Tensor(np.exp(*args, **kwargs))
  @staticmethod
  def max(*args, **kwargs): return Tensor(np.max(*args, **kwargs))
  @staticmethod
  def sum(*args, **kwargs): return Tensor(np.sum(*args, **kwargs))
  @staticmethod
  def hstack(*args, **kwargs): return Tensor(np.hstack(*args, **kwargs))

  def backward(self):
    def toposort(node, visited, ret):
      visited.add(node)
      for t in node.ctx.parents:
        if t not in visited and t.ctx: toposort(t, visited, ret)
      ret.append(node)
      return ret
    self.grad = np.array(1)
    for node in reversed(toposort(self, set(), [])):
      for t, g in zip(node.ctx.parents, node.ctx.backward(node.grad)):
        t.grad = g if t.grad is None else t.grad + g

class Function:
  def __init__(self, *x): self.parents = x

  @classmethod
  def apply(fn, *x): ctx = fn(*x); return Tensor(ctx.forward(*[t.data for t in x]), ctx=ctx)

  def forward(self, *args): raise NotImplementedError
  def backward(self, *args): raise NotImplementedError

class Neg(Function):
  def forward(self, x): return -x

class Add(Function):
  def forward(self, x, y): return x + y
  def backward(self, grad): return grad, grad

class Sub(Function):
  def forward(self, x, y): return x - y

class Mul(Function):
  def forward(self, x, y): self.x, self.y = x, y; return x * y
  def backward(self, grad): return self.y * grad, self.x * grad

class Pow(Function):
  def forward(self, x, y): return x ** y

class Div(Function):
  def forward(self, x, y): return x / y

class MatMul(Function):
  def forward(self, x, y): return x @ y