import torch
import torch.nn as nn
from torch.nn import functional as F

# The mathematical trick in self attention


# previously tokens were not talking to each other but now we want the token at ith index to talk to
# all tokens at indices < i
# the simplest way for the tokens to communicate is to take the average of preceding tokens i.e. take all channels
# at i plus all preceding i, average those and get a feature vector that will summarize i in the context of its history
# although this is a weak form of interaction, lossy, losing ton of info like spatial arrangement of these tokens etc
torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)

# We want x[b,t] = mean_{i<=t} x[b,i]
# bow = batch of words
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (T,C)
        xbow[b,t] = torch.mean(xprev, 0)
print(x[0])
print(xbow[0])

# --------------------------------------


# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.ones(3,3)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print("a :\n",a)
print("b :\n",b)
print("c :\n",c)

# wt if we use lower triangular matrix
a = torch.tril(torch.ones(3,3))
c = a @ b
print("a :\n",a)
print("c :\n",c)

# we got the sum across rows, now to get the average we need to update our matrix a
a = a / torch.sum(a, 1, keepdim=True)
c = a @ b
print("a :\n",a)
print("c :\n",c)

# --------------------------------------
# back to xbow

wei = torch.tril(torch.ones(T,T))
wei = wei / torch.sum(wei, 1, keepdim=True)
print("wei :\n", wei)
xbow2 = wei @ x # TT X BTC becomes BTT X BTC  ---> BTC
# unction in PyTorch used to check if all elements in two tensors are close within a certain tolerance. 
torch.allclose(xbow, xbow2)

# another way is using softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
print("wei :\n", wei)
wei = F.softmax(wei, dim=-1) # softmax - exponentiate every element and divide by the sum of those elements
print("wei :\n", wei)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

# self attention
# instead of taking a uniform vector of 0's we should choose data depedent values as tokens are more or less dependent
# on different other tokens
# every single token will emit 2 vectors - query(wt am i looking for), key(wt do i contain), and to get affinities between
# token we do the dot product of my query vec and key vec of all other tokens
B,T,C = 4,8,32
x = torch.randn(B,T,C)
# a single Head performing self-attention
head_size = 16 # Specifies the dimensionality of the query and key vectors for the attention mechanism.
# The model wants to make tokens interact with each other. To do this, it creates two sets of smaller representations of each word:
# key: It's like summarizing each word's information into a smaller space (represented by head_size = 16). Think of this as capturing key characteristics of each word.
# query: Similar to key, but this summarizes different aspects of each word in another smaller space.
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1) # how much info to aggregate from tokens in the past
# x is like private information of the token, v is the thing that gets aggregated for the purposes
# of single head between different nodes
v = value(x)
out = wei @ v

# NOTES
# Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
# There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
# Each example across batch dimension is of course processed completely independently and never "talk" to each other
# In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
# "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
# "Scaled" attention additional divides wei by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much. Illustration below

k = torch.randn(B,T,head_size)
q = torch.randn(B,T,head_size)
wei = q @ k.transpose(-2, -1) * head_size**-0.5 # without this the variance of wei will be of the order of headsize
print(k.var())
print(q.var())
print(wei.var()) 
# wei is fed to softmax to it should be fairly diffused especially during initialization

print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)) # here we get diffused values out of softmax
# tensor([0.1925, 0.1426, 0.2351, 0.1426, 0.2872])
print(torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5])*8, dim=-1)) # gets too peaky, converges to one-hot
# so for initialization we do not want extreme values
# tensor([0.0326, 0.0030, 0.1615, 0.0030, 0.8000])