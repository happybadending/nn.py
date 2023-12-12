def progressed(it):
  n = len(it)
  for i, k in enumerate(it, 1):
    yield k
    print(end=f'\r{i / n * 100:3.0f}%%|%-65s| {i}/{n}' % (chr(9608) * (65 * i // n)))
  print()

class Tensor:
  def __init__(self, data, ctx=None):
    self.data = self.numpy(data)
    self.grad = None
    self.ctx = ctx

  def __repr__(self): return f'{self.data!r}{f' grad= {self.grad!r}' if self.grad else ''}'
  def __getitem__(self, x): return Tensor(self.data.__getitem__(x))
  @property
  def shape(self): return self.data.shape

  @staticmethod
  def numpy(x):
    import numpy
    return numpy.array(x, dtype=x.dtype if getattr(x, 'dtype', False) else numpy.float32)

  @staticmethod
  def tri(x, lower=1, upper=0):
    y = Tensor(range(x)).reshape(1, x).expand(x, x)
    return (y <= y.T) * lower + (y > y.T) * upper

  #def split(self, n): splits on axis=-1 (make general or chop?)
  #  dxd= list(self.reshape(self.shape[0], n, self.shape[1] // n).transpose(1, 0, 2))
  #dxdxd= list(self.reshape(self.shape[0], self.shape[1], n, self.shape[2] // n).transpose(2, 0, 1, 3))

  @property
  def T(self):#return self.transpose(*reversed(range(len(self.shape))))
    axis = range(len(self.shape))
    return self.transpose(*axis[:-2], axis[-1], axis[-2]) if len(axis) > 1 else self.transpose(*axis)
  def transpose(self, *axis): return Transpose.apply(self, axis=axis)
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
  def mean(self, axis=-1, keepdim=True): return self.sum(axis=axis, keepdim=keepdim) / self.shape[axis]
  def softmax(self):
    y = (self - self.max()).exp()
    return y / y.sum()

  def exp(self): return Exp.apply(self)
  def __matmul__(self, x):
    m = min(len(self.shape) - 1, len(x.shape) - 1, 1)
    y = self.reshape(*self.shape[:-1], *(1,) * m, self.shape[-1])
    z = x.reshape(*x.shape[:-2], *(1,) * m, *x.shape[-min(len(x.shape), 2):]).T
    return (y * z).sum()

  def tanh(self): 2 / (1 + (-2 * self).exp()) - 1
  def gelu(self): 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())

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
  def apply(fn, *x, **kwargs): ctx = fn(*x); return Tensor(ctx.forward(*[t.data for t in x], **kwargs), ctx=ctx)

  def forward(self, *args): raise NotImplementedError
  def backward(self, *args): raise NotImplementedError

class Transpose(Function):
  def forward(self, x, axis): return x.transpose(axis)

class Reshape(Function):
  def forward(self, x, shape): return x.reshape(shape)

class Expand(Function):
  def forward(self, x, shape): import numpy; return numpy.broadcast_to(x, shape)

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

class Less(Function):
  def forward(self, x, y): return x < y

class Sum(Function):
  def forward(self, x, axis, keepdim): return x.sum(axis=axis, keepdims=keepdim)

class Max(Function):
  def forward(self, x, axis, keepdim): return x.max(axis=axis, keepdims=keepdim)

class Exp(Function):
  def forward(self, x): import numpy; return numpy.exp(x)