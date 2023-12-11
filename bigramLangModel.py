import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # independent sequences processed in parallel
block_size = 8 # maximum context length for predictions
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu' # run on a GPU if u have it
eval_iters = 200
embedding_dim = 64
n_head = 4
n_layer = 4
dropout = 0.0

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# find all unique characters
tot_chars = sorted(list(set(text)))
# print(tot_chars)

vocab_size = len(tot_chars)
# print(vocab_size)

# map characters to integers
stoi = {c:i for i,c in enumerate(tot_chars)}
itos = {i:c for i,c in enumerate(tot_chars)}
# lambda fxn to encode string to int
encode = lambda s : [stoi[c] for c in s]
# decode - int list to string
decode = lambda ls : ''.join([itos[i] for i in ls])

# encode the entire data and store it in torch.tensor
data = torch.tensor(encode(text))
# split the data into 90% training set and 10% validation set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # move the loaded data to device
    x, y = x.to(device), y.to(device)
    return x, y

# @torch.no_grad() is a decorator that decorates the estimate_loss() function.
# When you call this decorated function, any operations performed inside estimate_loss() won't track gradients.
# This is useful during evaluation or inference because you typically don't need to compute gradients during these phases, thus saving memory and computation.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
# multiple attentions in parallel and concatenating the results
# it helps to create multiple independent channels of communication, gather different types of data
# and decode the output
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Initialization (__init__):

# embedding_dim: Specifies the dimensionality of the input and output embeddings or features.
# Network Structure (self.net):

# nn.Sequential: Groups together a sequence of neural network layers to create the feedforward network.
# nn.Linear(embedding_dim, 4 * embedding_dim): The first linear layer performs an affine transformation,
#  projecting the input embedding_dim to a higher-dimensional space (4 * embedding_dim). This expansion allows the model to capture more complex patterns.
# nn.ReLU(): The rectified linear unit (ReLU) activation function introduces non-linearity by outputting the maximum between zero and the input value, 
# helping the network model more complex relationships in the data.
# nn.Linear(4 * embedding_dim, embedding_dim): The second linear layer reduces the dimensionality back to the original embedding_dim, compressing the 
# information learned in the higher-dimensional space.
# nn.Dropout(dropout): Applies dropout regularization, randomly setting a fraction of input units to zero during training to prevent overfitting. 
# The dropout value needs to be defined during initialization.
# Forward Pass (forward method):

# forward(self, x): Performs the forward pass through the defined network.
# self.net(x): Passes the input x through the sequential network structure, executing the linear transformations and activation functions in sequence.
# Returns the output of the network.

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

# Embedding Lookup (self.token_embedding_table):

# Imagine a table that assigns a unique set of numbers (vectors) to each word in a vocabulary. For example, 'dog' might be represented by [0.1, 0.3, -0.2], 'cat' by [0.2, -0.1, 0.5], and so on. These numbers are called embeddings.
# When you give this model a word, it looks up the word's embedding in this table. So, 'dog' might correspond to [0.1, 0.3, -0.2] in the table.
# Prediction Layer (self.lm_head):

# This part takes the embeddings and tries to predict the probabilities or likelihood of the next word. It's like asking the model, "Given the word 'dog,' what's the chance that the next word is 'runs,' 'barks,' or any other word in the vocabulary?"
# It uses these embeddings to calculate the chances for all words in the vocabulary, giving a score to each word.
# Putting It Together (forward method):

# When you give this model a sequence of words (or tokens), it looks up the embeddings for each word in the sequence.
# Then, it predicts the likelihood of the next word(s) based on these embeddings.
# This process is repeated for each word in the sequence to predict what might come next after each word.
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        # encoding the position of tokens as well
        self.position_embedding_table = nn.Embedding(block_size, embedding_dim)
        # self.sa_head = Head(embedding_dim)
        self.sa_heads = MultiHeadAttention(4, embedding_dim//4) # 4 head of 8-dim self attention
        self.ffwd = FeedFoward(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, b_t_input, targets=None):
        B, T = b_t_input.shape
        tok_emb = self.token_embedding_table(b_t_input) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.sa_heads(x) # apply one head of self attention BTC
        x = self.ffwd(x) # BTC
        # self attention is gathering all the data and in feed fwd, each token will process the data individually
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, b_t_input, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            b_t_input_cond = b_t_input[:, -block_size:]
            logits, loss = self(b_t_input_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            b_t_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            b_t_input = torch.cat((b_t_input, b_t_next), dim=1) # (B, T+1)
        return b_t_input

model = BigramLanguageModel()
# move the model params to device
m = model.to(device)

# PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))