from  nn import Tensor
import os, urllib.request, argparse
import tiktoken
from torch import load

class Embedding():
  def __init__(self): self.weight = None
  def __call__(self, x): return self.weight[x.data.astype(int)]

class Linear:
  def __init__(self): self.weight, self.bias = None, None
  def __call__(self, x): return x @ self.weight + self.bias

class LayerNorm():
  def __init__(self): self.weight, self.bias = None, None
  def __call__(self, x, eps=1e-5):
    rms = ((x * x).mean() + eps) ** 0.5
    return x / rms * self.weight + self.bias

class FeedForward():
  def __init__(self): self.c_fc, self.c_proj = Linear(), Linear()
  def __call__(self, x): return self.c_proj(self.c_fc(x).gelu())

class Attention():
  def __init__(self, n_heads):
    self.n_heads = n_heads
    self.bias = None
    self.c_attn = Linear()
    self.c_proj = Linear()

  def __call__(self, x):
    mask = Tensor(range(seq := x.shape[0])).reshape(1, seq).expand(seq, seq)
    mask = (mask > mask.transpose(1, 0)) * -1e10
    q, k, v = self.c_attn(x).reshape(seq, 3, self.n_heads, 64).transpose(1, 2, 0, 3)
    return self.c_proj(((q @ k.transpose(0, 2, 1) / q.shape[-1] ** 0.5 + mask).softmax() @ v).transpose(1, 0, 2).reshape(*x.shape))

class TransformerBlock():
  def __init__(self, n_heads):
    self.attn = Attention(n_heads)
    self.mlp = FeedForward()
    self.ln_1 = LayerNorm()
    self.ln_2 = LayerNorm()

  def __call__(self, x):
    y = x + self.attn(self.ln_1(x))
    return y + self.mlp(self.ln_2(y))

class Transformer():
  def __init__(self, params):
    self.wte = Embedding()
    self.wpe = Embedding()
    self.block = [TransformerBlock(params['n_heads']) for _ in range(params['n_layers'])]
    self.ln_f = LayerNorm()

  def __call__(self, x, temperature):
    y = self.wte(x) + self.wpe(Tensor(range(len(x.data))))
    for h in self.block: y = h(y)
    y = self.ln_f(y) @ self.wte.weight.transpose(1, 0)
    return (y[-1] / (temperature + 1e-10)).softmax()

class GPT2:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer
  
  @staticmethod
  def progressed(it):
    n = len(it)
    for i, k in enumerate(it, 1):
      yield k
      print(end=f'\r{i / n * 100:3.0f}%%|%-65s| {i}/{n}' % (chr(9608) * (65 * i // n)))
    print()

  def generate(self, prompt, n_toks, temperature):
    toks = self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
    for _ in self.progressed(range(n_toks)):
      logits = self.model(Tensor(toks), temperature)
      toks.append(max(zip(logits.data, range(logits.shape[0])))[1])
    print(prompt, self.tokenizer.decode(toks[len(toks) - n_toks:]), sep='')

  @staticmethod
  def build(model_size):
    params = {
      'gpt2': dict(n_layers=12, n_heads=12),
      'gpt2-medium': dict(n_layers=24, n_heads=16),
      'gpt2-large': dict(n_layers=36, n_heads=20),
      'gpt2-xl': dict(n_layers=48, n_heads=25),
    }[model_size]
    if not os.path.exists(model_size+'.bin'): urllib.request.urlretrieve('https://huggingface.co/'+model_size+'/resolve/main/pytorch_model.bin', model_size+'.bin')
    weights = dict(block=[{} for _ in range(params['n_layers'])])
    def insert(d, k, v):
      if not k: return v
      if k[0] not in d: d[k[0]] = {}
      d[k[0]] = insert(d[k[0]], k[1:], v)
      return d
    for k, v in load(model_size+'.bin').items():
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

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_size', type=str, default='gpt2', help='[gpt2, gpt2-medium, gpt2-large, gpt2-xl]')
parser.add_argument('--prompt', type=str, default='Alan Turing theorized that computers would one day become', help='starting phrase')
parser.add_argument('--n_toks', type=int, default=8, help='tokens to generate')
parser.add_argument('--temperature', type=float, default=0, help='randomness')
args = parser.parse_args()

gpt = GPT2.build(args.model_size)
gpt.generate(args.prompt, args.n_toks, args.temperature)