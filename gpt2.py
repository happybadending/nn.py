from  nn import Tensor, progressed
import os, urllib.request
import tiktoken
import torch
import fire

class Embedding():
  def __init__(self): self.weight = None
  def __call__(self, x): return self.weight[x.data.astype(int)]

class Linear:
  def __init__(self): self.weight, self.bias = None, None
  def __call__(self, x): return x @ self.weight + self.bias

class LayerNorm():
  def __init__(self): self.weight, self.bias = None, None
  def __call__(self, x, eps=1e-5):
    rms = ((x*x).mean() + eps) ** 0.5
    return x / rms * self.weight + self.bias

class FeedForward():
  def __init__(self): self.c_fc, self.c_proj = Linear(), Linear()
  def __call__(self, x):  return self.c_proj(self.c_fc(x).gelu())

class Attention():
  def __init__(self):
    self.bias = None  
    self.c_attn = Linear()
    self.c_proj = Linear()

  def __call__(self, x, n_heads):
    def attention(q, k, v, mask): return (q @ k.T / q.shape[-1] ** 0.5 + mask).softmax() @ v
    mask = Tensor(range(x.shape[0])).reshape(1, x.shape[0]).expand(x.shape[0], x.shape[0])
    mask = (mask > mask.T) * -1e10
    y = self.c_attn(x)
    qkv = list(y.reshape(y.shape[0], 3, y.shape[1] // 3).transpose(1, 0, 2))
    qkv = list(map(lambda x: x.reshape(x.shape[0], n_heads, x.shape[1] // n_heads).transpose(1, 0, 2), qkv))
    y = [attention(q, k, v, mask) for q, k, v in zip(*qkv)]
    return self.c_proj(Tensor([z.data for z in y]).transpose(1, 0, 2).reshape(*x.shape))

class TransformerBlock():
  def __init__(self):
    self.attn = Attention()
    self.mlp = FeedForward()
    self.ln_1 = LayerNorm()
    self.ln_2 = LayerNorm()

  def __call__(self, x, n_heads):
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
    y = self.wte(x) + self.wpe(Tensor(range(len(x.data))))
    for h in self.block: y = h(y, self.p['n_heads'])
    y = self.ln_f(y) @ self.wte.weight.T
    return (y / (temperature + 1e-10)).softmax()

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
    if not os.path.exists(model_size+'.bin'):
      urllib.request.urlretrieve('https://huggingface.co/'+model_size+'/resolve/main/pytorch_model.bin', model_size+'.bin')
    weights = dict(block=[{} for _ in range(params['n_layers'])])
    def insert(d, k, v):
      if not k: return v
      if k[0] not in d: d[k[0]] = {}
      d[k[0]] = insert(d[k[0]], k[1:], v) 
      return d
    for k, v in torch.load(model_size+'.bin').items():
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
          setattr(cls, k, Tensor(v.numpy()))
    load_weights(gpt.model, weights)
    return gpt

gpt = GPT2.build('gpt2')
fire.Fire(gpt.generate)