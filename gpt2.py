from  nn import Tensor
import os, urllib.request, random, argparse
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
    self.kv_cache = Tensor([None, None])

  def __call__(self, x):
    mask = Tensor(range(seq := x.shape[0])).reshape(1, seq)
    mask = (mask > mask.transpose(1, 0)) * -1e10
    q, k, v = self.c_attn(x).reshape(seq, 3, self.n_heads, 64).transpose(1, 0, 2, 3)
    self.kv_cache = Tensor([[y.data for y in list(self.kv_cache[0]) + list(k)], [y.data for y in list(self.kv_cache[1]) + list(v)]])
    q, k, v = q.transpose(1, 0, 2), self.kv_cache[0].transpose(1, 2, 0), self.kv_cache[1].transpose(1, 0, 2)
    return self.c_proj(((q @ k / q.shape[-1] ** 0.5 + mask).softmax() @ v).transpose(1, 0, 2).reshape(*x.shape))

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
    self.h = [TransformerBlock(params['n_heads']) for _ in range(params['n_layers'])]
    self.ln_f = LayerNorm()

  def __call__(self, x, temperature, start_pos, i):
    pos = Tensor(range(start_pos) if not i else [start_pos + i - 1])
    y = self.wte(x) + self.wpe(pos)
    for block in self.h: y = block(y)
    y = self.ln_f(y) @ self.wte.weight.transpose(1, 0)
    return (y[-1] / (temperature + 1e-10)).softmax()

class GPT2:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def generate(self, prompt, n_toks, temperature):
    print(f'\033[32m{prompt}\033[0m', end='')
    toks = self.tokenizer.encode(prompt, allowed_special={'<|endoftext|>'})
    start_pos = len(toks)
    for i in range(n_toks):
      logits = self.model(Tensor(toks), temperature, start_pos, i)
      toks = random.choices(range(50257), logits.data)
      print(self.tokenizer.decode(toks), end='', flush=True)
    print()

  @staticmethod
  def build(model_size):
    params = {
      'gpt2': dict(n_layers=12, n_heads=12),
      'gpt2-medium': dict(n_layers=24, n_heads=16),
      'gpt2-large': dict(n_layers=36, n_heads=20),
      'gpt2-xl': dict(n_layers=48, n_heads=25),
    }[model_size]
    gpt = GPT2(Transformer(params), tiktoken.get_encoding('gpt2'))
    if not os.path.exists(model_size+'.bin'): urllib.request.urlretrieve('https://huggingface.co/'+model_size+'/resolve/main/pytorch_model.bin', model_size+'.bin')
    for k, v in load(model_size+'.bin').items():
      k, attr = k.rsplit('.', 1)
      dest = gpt.model
      for p in k.split('.'): dest = dest[int(p)] if p.isdigit() else getattr(dest, p)
      setattr(dest, attr, Tensor(v.numpy()))
    return gpt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model_size', type=str, default='gpt2', help='[gpt2, gpt2-medium, gpt2-large, gpt2-xl]')
parser.add_argument('-p', '--prompt', type=str, default='Alan Turing theorized that computers would one day become', help='starting phrase')
parser.add_argument('-n', '--n_toks', type=int, default=8, help='tokens to generate')
parser.add_argument('-t', '--temperature', type=float, default=0, help='randomness')
args = parser.parse_args()

gpt = GPT2.build(args.model_size)
gpt.generate(args.prompt, args.n_toks, args.temperature)