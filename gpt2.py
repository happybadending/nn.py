# based on https://github.com/jaymody/picoGPT
# TODO: move stuff into nn.py and remove numpy dependency
# TODO: prettier load weights

import os, urllib.request
import tiktoken
import safetensors.numpy
import numpy as np
from tqdm import tqdm
import fire

class Embedding():
  def __init__(self, weights):
    self.weight = weights['weight']

  def __call__(self, x):
    return self.weight[x]

class Linear:
  def __init__(self, weights):
    self.weight = weights['weight']
    self.bias = weights['bias']

  def __call__(self, x):
    return x @ self.weight + self.bias

class Attention():
  def __init__(self, weights):
    self.c_attn = Linear(weights['c_attn'])
    self.c_proj = Linear(weights['c_proj'])

  def __call__(self, x, n_heads):
    def softmax(x):
      e = np.exp(x - np.max(x, axis=-1, keepdims=True))
      return e / np.sum(e, axis=-1, keepdims=True)
    def attention(q, k, v, mask):
      return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
    mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    y = self.c_attn(x)
    qkv = np.split(y, 3, axis=-1) # TODO: add kv cache
    qkv = list(map(lambda x: np.split(x, n_heads, axis=-1), qkv))
    y = [attention(q, k, v, mask) for q, k, v in zip(*qkv)]
    return self.c_proj(np.hstack(y))

class FeedForward():
  def __init__(self, weights):
    self.c_fc = Linear(weights['c_fc'])
    self.c_proj = Linear(weights['c_proj'])

  def __call__(self, x):
    def gelu(x):
      return 0.5 * x * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
    return self.c_proj(gelu(self.c_fc(x)))

class LayerNorm():
  def __init__(self, weights):
    self.weight = weights['weight']
    self.bias = weights['bias']

  def __call__(self, x, eps=1e-5):
    rms = ((x*x).mean(axis=-1, keepdims=True) + eps) ** 0.5
    return x / rms * self.weight + self.bias

class TransformerBlock():
  def __init__(self, weights):
    self.attn = Attention(weights['attn'])
    self.mlp = FeedForward(weights['mlp'])
    self.ln_1 = LayerNorm(weights['ln_1'])
    self.ln_2 = LayerNorm(weights['ln_2'])

  def __call__(self, x, n_heads):
    y = x + self.attn(self.ln_1(x), n_heads)
    return y + self.mlp(self.ln_2(y))

class Transformer():
  def __init__(self, weights, params):
    self.p = params
    self.wte = Embedding(weights['wte'])
    self.wpe = Embedding(weights['wpe'])
    self.h = [TransformerBlock(weights['block'][i]) for i in range(self.p['n_layers'])]
    self.ln_f = LayerNorm(weights['ln_f'])

  def __call__(self, x, temperature):
    def softmax(x):
      e = np.exp(x - np.max(x, axis=-1, keepdims=True))
      return e / np.sum(e, axis=-1, keepdims=True)
    y = self.wte(x) + self.wpe(range(len(x)))
    for h in self.h: y = h(y, self.p['n_heads'])
    y = self.ln_f(y) @ self.wte.weight.T
    return softmax(y / (temperature + 1e-10))

class GPT2:
  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  def generate(self, prompt, temperature=0, n_toks=50):
    toks = self.tokenizer.encode(prompt)
    for _ in tqdm(range(n_toks), 'cooking'):
      logits = self.model(toks, temperature)
      toks.append(int(np.argmax(logits[-1]))) # TODO: not be greedy
    print(self.tokenizer.decode(toks[len(toks) - n_toks:]))

  @staticmethod
  def build(model_size):
    params = {
      'gpt2': dict(n_layers=12, n_heads=12, dim=768, n_vocab=50257, eps=1e-5),
      # TODO: maybe add more sizes
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
    return GPT2(Transformer(weights, params), tiktoken.get_encoding(model_size))

gpt = GPT2.build('gpt2')
fire.Fire(gpt.generate)