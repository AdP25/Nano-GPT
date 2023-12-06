import torch
import torch.nn as nn
from torch.nn import functional as F

with open("input.txt", 'r', encoding='utf-8') as file:
    text = file.read()

# print("Total chars in file : ", len(text))

print(text[:100])
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
print(encode("It's me Aditi!"))
print(decode(encode("It's me Aditi!")))

# encode the entire data and store it in torch.tensor
data = torch.tensor(encode(text))
print(data.shape, data.dtype, data.device)
print(data[:50])

# split the data into 90% training set and 10% validation set
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# we do not feed entire text to the transformer all at once
# sample random little chunks out of the data set of length typically 8
block_size = 8
train_data_test = train_data[:block_size+1] # +1 because in a chunk of 9(8+1) chars,
# there are 8 individual examples packed

for i in range(block_size):
    context = train_data_test[:i+1]
    target = train_data_test[i+1]
    print(f"INPUT : {context}, OUTPUT : {target}")

# BATCH DIMENSION
# batches of multiple chunks stacked up in a single tensor will be processes parallely 
# by the GPUs

# In PyTorch, if you create a neural network and initialize its weights randomly without setting a seed, 
# each time you run the code, the weights will be initialized differently. 
# However, by setting the seed, the weights will always be initialized in the same way, making your experiments reproducible (1337 - LEET :P)
torch.manual_seed(1337)
# no of independent sequences to be processed in parallel
batch_size = 4
# max context length for prediction
block_size = 8

def get_batch(data_type):
    if(data_type == "train"):
        data = train_data
    else:
        data = val_data
    # torch.randint() is a PyTorch function that generates random integers within a specified range.
    # len(data) - block_size defines the upper limit of the range from which random integers will be generated
    # (batch_size,) specifies the shape of the output tensor. The generated random integers will be organized into a tensor of shape (batch_size,), 
    # where batch_size is a variable representing the number of random indices to generate.
    rdm_data = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in rdm_data])
    y = torch.stack([data[i+1:i+block_size+1] for i in rdm_data])
    return x,y

x_batch, y_batch = get_batch("train")
print('X_BATCH :')
print(x_batch.shape)
print(x_batch) # input to the transformer
print('Y_BATCH :')
print(y_batch)

for i in range(batch_size):
    for j in range(block_size):
        context = x_batch[i, :j+1]
        target = y_batch[i,j]
        print(f"For Input = {context.tolist()} Target = {target}")

embedding_dim = vocab_size
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # the input and output size of the embedding table are both set to vocab_size
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
    # The forward method in a neural network class defines the computation performed when input data is passed through the network.
    # It describes how the input data flows through the layers of the network to produce the output (predictions or activations).
    def forward(self, b_t_input, targets=None):
        # Batch(4) Time(8) Channel(65) - BTC
        # b_t_input and targets are both (B,T) tensor of integers
        # b_t_input represents a batch of sequences (tensor of integers) for which we want to predict the next tokens.
        # logits computes the embeddings for the input sequences using the embedding table.
        logits = self.embedding_table(b_t_input) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            # in case of multi-dimensional logits, cross entropy expects B C T not B T C hence reshaping
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    # B T -> B T+1 -> B T+2 ... upto max_new_tokens
    def generate(self, b_t_input, max_new_tokens):
        # b_t_input is (B, T) array of indices
        for _ in range(max_new_tokens):
            logits, loss = self(b_t_input) # BTC
            # focus only on the last time step
            # in the context of sequence data (like text or time-series), the last time step refers to the final element in the sequence.
            # This extraction is often useful in tasks where the model needs to make predictions 
            # or decisions based on the last observed data point or context in each sequence.
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample a new token index based on the predicted probabilities using torch.multinomial.
            b_t_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append the sampled token index to the running sequence
            b_t_input = torch.cat((b_t_input, b_t_next), dim=1) # (B, T+1)
        return b_t_input

m = BigramLanguageModel(vocab_size)
logits, loss = m(x_batch, y_batch)
print(logits.shape)
# expected ->{ -ln(1/65) = 4.17438 }
print(loss)




