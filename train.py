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
print(data[:100])

# split the data into 90% training set and 10% validation set
n = int(0.9 * len(data))
train_set = data[:n]
val_set = data[n:]



