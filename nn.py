import numpy as np

class Tensor:
  def __init__(self, data, ctx=None):
    self.data = np.array(data)
    self.grad = None
    self.ctx = ctx

  def __repr__(self):
    return f"<T {self.data}>"

  def __neg__(self): return Neg.apply(self)

  def __add__(self, x): return Add.apply(self, x)
  def __sub__(self, x): return Sub.apply(self, x)
  def __mul__(self, x): return Mul.apply(self, x)
  def __pow__(self, x): return Pow.apply(self, x)
  def __truediv__(self, x): return Div.apply(self, x)
  def __matmul__(self, x): return MatMul.apply(self, x)

  def backward(self):
    def toposort(node, visited, ret):
      visited.add(node)
      if node.ctx is None:
        for t in node.ctx.parents:
          if t not in visited: toposort(t, visited, ret)
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