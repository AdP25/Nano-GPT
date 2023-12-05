import torch

with open("input.txt", 'r', encoding='utf-8') as file:
    text = file.read()

# print("Total chars in file : ", len(text))

print(text[:100])
# find all unique characters
tot_chars = sorted(list(set(text)))
# print(tot_chars)

# len_chars = len(tot_chars)
# print(len_chars)

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



