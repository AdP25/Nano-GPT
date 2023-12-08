import torch

# The mathematical trick in self attention


# previously tokens were not talking to each other but now we want the token at ith index to talk to
# all tokens at indices < i
# the simplest way for the tokens to communicate is to take the average of preceding tokens i.e. take all channels
# at i plus all preceding i, average those and get a feature vector that will summarize i in the context of its history
# although this is a weak form of interaction, lossy, losing ton of info like spatial arrangement of these tokens etc
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


