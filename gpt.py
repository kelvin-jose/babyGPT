import torch
import random
random.seed(0)
torch.manual_seed(0)

INPUT_FILE_LOCATION = 'data/input.txt'
input_file = open(INPUT_FILE_LOCATION).read()
input_file_len = len(input_file)

vocab = sorted(set(input_file))
train_ratio = 0.7
Xtrain, Xtest = input_file[:int(input_file_len * train_ratio)], input_file[int(input_file_len * train_ratio):]

block_size = 8
char2id = {char: i for i, char in enumerate(vocab)}
id2char = {i: char for char, i in char2id.items()}

encode = lambda chars: [char2id[char] for char in chars]
decode = lambda ids: [id2char[i] for i in ids]
decode(encode('kelvin'))

Xtrain_enc, Xtest_enc = torch.tensor(encode(Xtrain)), torch.tensor(encode(Xtest))

# batch generator
def get_random_batch(choice = 'train', batch_size = 1):
    if choice == 'train':
        idxs = random.sample(range(0, len(Xtrain) - block_size), batch_size)
        Xbatch = torch.stack([Xtrain_enc[idx:idx + block_size] for idx in idxs])
        Ybatch = torch.stack([Xtrain_enc[idx + 1:idx + block_size + 1] for idx in idxs])
        return Xbatch, Ybatch
    elif choice == 'test':
        idxs = random.sample(range(0, len(Xtest)), batch_size)
        Xbatch = torch.stack([Xtest_enc[idx:idx + block_size] for idx in idxs])
        Ybatch = torch.stack([Xtest_enc[idx + 1:idx + block_size + 1] for idx in idxs])
        return Xbatch, Ybatch

batch_size = 4
Xbatch, Ybatch = get_random_batch('train', 4)

# Masked Head model definition
class Head(torch.nn.Module):
    def __init__(self, token_dim, ndim):
        super().__init__()
        self.to_query = torch.nn.Linear(token_dim, ndim)
        self.to_key = torch.nn.Linear(token_dim, ndim)
        self.to_value = torch.nn.Linear(token_dim, ndim)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        # B, T, ndim
        query = self.to_query(x)
        key = self.to_key(x)
        dp = query @ key.transpose(-2, -1) * (key.shape[-1]**-0.5)
        _, T, _ = x.shape
        dp = dp.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        sm = torch.functional.F.softmax(dp, -1) # B, T, T
        value = self.to_value(x) # B, T, C
        wembs = sm @ value
        return wembs

class FeedForward(torch.nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.linear1 = torch.nn.Linear(_in, _in * 2)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(_in * 2, _out)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# MultiHead
class MultiHeadMaskedAttention(torch.nn.Module):
    def __init__(self, token_dim, nheads):
        super().__init__()
        self.heads = [Head(token_dim, token_dim // nheads) for head in range(nheads)]
        self.ffwd = FeedForward(token_dim, token_dim)
    
    def forward(self, x):
        return self.ffwd(torch.cat([head(x) for head in self.heads], dim = -1))

# Block    
class Block(torch.nn.Module):
    def __init__(self, token_dim, nheads):
        super().__init__()
        self.mheads = MultiHeadMaskedAttention(token_dim, nheads)
        
    def forward(self, x):
        output = self.mheads(x)
        return output
    
# GPT starter code
class babyGPT(torch.nn.Module):
    def __init__(self, token_dim, nheads, nblocks):
        super().__init__()
        self.token_embeds = torch.nn.Embedding(len(vocab), token_dim)
        self.pos_embeds = torch.nn.Embedding(len(vocab), token_dim)
        self.blocks = [Block(token_dim, nheads) for block in range(nblocks)]
        self.linear = torch.nn.Linear(token_dim, len(vocab))
        
    def forward(self, x, test = False):
        t_embeds = self.token_embeds(x)
        _, T = x.shape
        p_embeds = self.pos_embeds(torch.arange(T))
        embeds = t_embeds + p_embeds
        for block in self.blocks:
            embeds = block(embeds)
        logits = self.linear(embeds)
        return logits
    
def generate(model, max_tokens):
    model.eval()
    tokens = torch.tensor([[0]])
    text = ''
    for t in range(max_tokens):
        logits = model(tokens)[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim = -1)
        idx_next = torch.multinomial(probs, num_samples=1) 
        text += decode(idx_next.tolist()[0])[0]
        tokens = torch.cat((tokens, idx_next), dim=1)
        tokens = tokens[:, -block_size:]
        print(text)
        

# basic training script
nheads = 4
nblocks = 2
lr = 0.001
token_dim = 32
batch_size = 64
train_steps = 50000

bgpt = babyGPT(token_dim, nheads, nblocks)
optim = torch.optim.AdamW(bgpt.parameters(), lr = lr)

for step in range(train_steps):
    xtrain, ytrain = get_random_batch('train', batch_size)
    logits = bgpt(xtrain)
    B, T, C = logits.shape
    loss = torch.nn.functional.cross_entropy(logits.reshape(B*T, C), ytrain.reshape(-1))
    optim.zero_grad()
    loss.backward()
    optim.step()
    if step % 100 == 0:
        print(f'step : {step} loss : {loss.detach():3f}')

generate(bgpt, 500)