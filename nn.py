import numpy as np

class Tensor:
  def __init__(self, data, ctx=None):
    self.data = self.numpy(data)
    self.grad = None
    self.ctx = ctx

  @property
  def shape(self): return self.data.shape
  def __getitem__(self, x): return Tensor(self.data[x])
  def __repr__(self): return f'{self.data!r}{f' grad= {self.grad!r}' if self.grad else ''}'

  @staticmethod
  def numpy(x): return np.array(x, dtype=x.dtype if getattr(x, 'dtype', False) else np.float32)

  def transpose(self, *shape): return Transpose.apply(self, shape=shape)
  def reshape(self, *shape): return Reshape.apply(self, shape=shape)
  def expand(self, *shape): return Expand.apply(self, shape=shape)

  def broadcasted(self, x, reflected=False):
    y, z = self, x if isinstance(x, Tensor) else Tensor(x)
    if reflected: y, z = z, y
    if (ys := y.shape) == (zs := z.shape): return y, z
    ds = len(ys) - len(zs)
    if ds < 0: y = y.reshape(*(1,) * -ds + ys)
    if ds > 0: z = z.reshape(*(1,) * ds + zs)
    if (ys := y.shape) == (zs := z.shape): return y, z
    rs = [max(d) for d in zip(ys, zs)]
    if ys != rs: y = y.expand(*rs)
    if zs != rs: z = z.expand(*rs)
    return y, z
  def __add__(self, x): return Add.apply(*self.broadcasted(x))
  def __sub__(self, x): return Sub.apply(*self.broadcasted(x))
  def __mul__(self, x): return Mul.apply(*self.broadcasted(x))
  def __pow__(self, x): return Pow.apply(*self.broadcasted(x))
  def __truediv__(self, x): return self * x ** -1
  def __radd__(self, x): return Add.apply(*self.broadcasted(x, reflected=True))
  def __rsub__(self, x): return Sub.apply(*self.broadcasted(x, reflected=True))
  def __rmul__(self, x): return Mul.apply(*self.broadcasted(x, reflected=True))
  def __rpow__(self, x): return Pow.apply(*self.broadcasted(x, reflected=True))
  def __rtruediv__(self, x):  return x * self ** -1
  def __lt__(self, x): return Less.apply(*self.broadcasted(x))
  def __gt__(self, x): return Less.apply(*self.broadcasted(x, reflected=True))
  def __le__(self, x): return 1 - (self > x)
  def __ge__(self, x): return 1 - (self < x)

  def sum(self, axis=-1, keepdim=True): return Sum.apply(self, axis=axis, keepdim=keepdim)
  def max(self, axis=-1, keepdim=True): return Max.apply(self, axis=axis, keepdim=keepdim)
  def mean(self, axis=-1, keepdim=True): return self.sum(axis, keepdim) / self.shape[axis]
  def softmax(self):
    y = 2 ** ((self - self.max()) * 1.44269504)
    return y / y.sum()

  #def __matmul__(self, x):
  #  n = min(len(self.shape) -1, (lxs := len(x.shape)) - 1, 1)
  #  y = self.reshape(*self.shape[:-1], *(1,) * n, self.shape[-1])
  #  z = x.reshape(*x.shape[:-2], *(1,) * n, *x.shape[-min(lxs, 2):])
  #  if len(rlzs := range(len(z.shape))) > 1: z = z.transpose(*rlzs[:-2], rlzs[-1], rlzs[-2])
  #  return (y * z).sum(keepdim=False)
  def __matmul__(self, x): return Tensor(np.matmul(self.data, x.data))

  def sigmoid(self): return 1 / (1 + 2 ** (self  * -1.44269504))
  def silu(self): return self * self.sigmoid()
  def tanh(self): return 2 * (2 * self).sigmoid() - 1
  def gelu(self): return self * (self * 1.702).sigmoid()

  def backward(self):
    def toposort(node, visited, ret):
      visited.add(node)
      for t in node.ctx.parents:
        if t not in visited and t.ctx: toposort(t, visited, ret)
      ret.append(node)
      return ret
    self.grad = self.numpy(1)
    for node in reversed(toposort(self, set(), [])):
      for t, g in zip(node.ctx.parents, node.ctx.backward(node.grad)):
        t.grad = t.grad + g if t.grad else g

class Function:
  def __init__(self, *x): self.parents = x

  @classmethod
  def apply(fn, *x, **kwargs):
    ctx = fn(*x)
    return Tensor(ctx.forward(*[t.data for t in x], **kwargs), ctx=ctx)

class Transpose(Function):
  def forward(self, x, shape): return x.transpose(shape)

class Reshape(Function):
  def forward(self, x, shape): return x.reshape(shape)

class Expand(Function):
  def forward(self, x, shape): return np.broadcast_to(x, shape)

class Add(Function):
  def forward(self, x, y): return x + y
  def backward(self, grad): return grad, grad

class Sub(Function):
  def forward(self, x, y): return x - y

class Mul(Function):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x * y
  def backward(self, grad): return self.y * grad, self.x * grad

class Pow(Function):
  def forward(self, x, y): return x ** y

class Less(Function):
  def forward(self, x, y): return x < y

class Sum(Function):
  def forward(self, x, axis, keepdim): return x.sum(axis, keepdims=keepdim)

class Max(Function):
  def forward(self, x, axis, keepdim): return x.max(axis, keepdims=keepdim)