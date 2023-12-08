from  nn import Tensor, progressed
import os, urllib.request
import tiktoken
import safetensors.numpy
import fire

class Embedding():
  def __init__(self): self.weight = None
  def __call__(self, x):
    assert isinstance(x, Tensor), type(x)
    assert not isinstance(x.data, Tensor), type(x.data)
    return self.weight[x.data]

class Linear:
  def __init__(self): self.weight = self.bias = None
  def __call__(self, x):
    assert isinstance(x, Tensor), type(x)
    assert not isinstance(x.data, Tensor), type(x.data)
    return x @ self.weight + self.bias

class Attention():
  def __init__(self):
    self.bias = None
    self.c_attn = Linear()
    self.c_proj = Linear()

  def __call__(self, x, n_heads):
    assert isinstance(x, Tensor), type(x)
    assert not isinstance(x.data, Tensor), type(x.data)
    def softmax(x):
      e = Tensor.exp((x - Tensor.max(x.data, axis=-1, keepdims=True)).data)
      return e / Tensor.sum(e.data, axis=-1, keepdims=True)
    def attention(q, k, v, mask):
      return softmax(q @ k.T / Tensor.sqrt(q.shape[-1]) + mask) @ v
    mask = (1 - Tensor.tri(x.shape[0], dtype=x.data.dtype)) * -1e10
    y = self.c_attn(x)
    qkv = Tensor.split(y.data, 3, axis=-1) # TODO: add kv cache
    qkv = list(map(lambda x: Tensor.split(x.data, n_heads, axis=-1), qkv.data))
    y = [attention(q, k, v, mask) for q, k, v in zip(*qkv)]
    for i in range(len(y)): y[i] = y[i].data
    return self.c_proj(Tensor.hstack(y))

class FeedForward():
  def __init__(self):
    self.c_fc = Linear()
    self.c_proj = Linear()

  def __call__(self, x):
    assert isinstance(x, Tensor), type(x)
    assert not isinstance(x.data, Tensor), type(x.data)
    def gelu(x):
      return 0.5 * x * (1 + Tensor.tanh(x.data * 0.7978845608 * (1 + 0.044715 * x.data * x.data)))
    return self.c_proj(gelu(self.c_fc(x)))

class LayerNorm():
  def __init__(self):
    self.weight = None
    self.bias = None

  def __call__(self, x, eps=1e-5):
    assert isinstance(x, Tensor), type(x)
    assert not isinstance(x.data, Tensor), type(x.data)
    rms = ((x*x).data.mean(axis=-1, keepdims=True) + eps) ** 0.5
    return x / rms * self.weight + self.bias

class TransformerBlock():
  def __init__(self):
    self.attn = Attention()
    self.mlp = FeedForward()
    self.ln_1 = LayerNorm()
    self.ln_2 = LayerNorm()

  def __call__(self, x, n_heads):
    assert isinstance(x, Tensor), type(x)
    assert not isinstance(x.data, Tensor), type(x.data)
    y = x + self.attn(self.ln_1(x), n_heads)
    return y + self.mlp(self.ln_2(y))

class Transformer():
  def __init__(self, params):
    self.p = params
    self.wte = Embedding()
    self.wpe = Embedding()
    self.block = [TransformerBlock() for _ in range(self.p['n_layers'])]
    self.ln_f = LayerNorm()

  def __call__(self, x, temperature):
    assert isinstance(x, Tensor), type(x)
    assert not isinstance(x.data, Tensor), type(x.data)
    def softmax(x):
      e = Tensor.exp((x - Tensor.max(x.data, axis=-1, keepdims=True)).data)
      return e / Tensor.sum(e.data, axis=-1, keepdims=True)
    y = self.wte(x) + self.wpe(Tensor(range(len(x.data))))
    for h in self.block: y = h(y, self.p['n_heads'])
    y = self.ln_f(y) @ self.wte.weight.T
    return softmax(y / (temperature + 1e-10))

class GPT2:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def generate(self, prompt='<|endoftext|>', n_toks=50, temperature=0):
    toks = self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
    for _ in progressed(range(n_toks)):
      logits = self.model(Tensor(toks), temperature)
      toks.append(int(Tensor.argmax(logits[-1].data).data)) # TODO: not be greedy
    print(self.tokenizer.decode(toks[len(toks) - n_toks:]))

  @staticmethod
  def build(model_size):
    params = {
      'gpt2': dict(n_layers=12, n_heads=12, dim=768, n_vocab=50257, eps=1e-5),
      'gpt2-medium': dict(n_layers=24, n_heads=16, dim=1024, n_vocab=50257, eps=1e-5),
      'gpt2-large': dict(n_layers=36, n_heads=20, dim=1280, n_vocab=50257, eps=1e-5),
      'gpt2-xl': dict(n_layers=48, n_heads=25, dim=1600, n_vocab=50257, eps=1e-5),
    }[model_size]
    if not os.path.exists(model_size+'.safetensors'): urllib.request.urlretrieve('https://huggingface.co/'+model_size+'/resolve/main/model.safetensors', model_size+'.safetensors')
    weights = dict(block=[{} for _ in range(params['n_layers'])])
    def insert(d, k, v):
      if not k: return v
      if k[0] not in d: d[k[0]] = {}
      d[k[0]] = insert(d[k[0]], k[1:], v)
      return d
    for k,v in safetensors.numpy.load_file(model_size+'.safetensors').items():
      p = k.split('.')
      if k.startswith('h'): insert(weights['block'][int(p[1])], p[2:], v)
      else: insert(weights, p, v)
    gpt = GPT2(Transformer(params), tiktoken.get_encoding('gpt2'))
    def load_weights(cls, w):
      for k,v in w.items():
        if isinstance(v, dict):
          load_weights(getattr(cls, k), v)
        elif isinstance(v, list):
          for i,d in enumerate(v):
            load_weights(getattr(cls, k)[i], d)
        else:
          setattr(cls, k, Tensor(v))
    load_weights(gpt.model, weights)
    return gpt

gpt = GPT2.build('gpt2')
fire.Fire(gpt.generate)