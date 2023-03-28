import torch
import random

INPUT_FILE_LOCATION = '../data/input.txt'
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
        idxs = random.sample(range(0, len(Xtrain)), batch_size)
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

# Masked Attention model definition
class MaskedAttention(torch.nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.to_query = torch.nn.Linear(64, ndim)
        self.to_key = torch.nn.Linear(64, ndim)
        self.to_value = torch.nn.Linear(64, ndim)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        # B, T, ndim
        query = self.to_query(x)
        key = self.to_key(x)
        dp = query @ key.transpose(-2, -1) * (key.shape[-1] ** -0.5)
        _, T, _ = x.shape
        dp = dp.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        sm = torch.functional.F.softmax(dp, -1) # B, T, T
        value = self.to_value(x) # B, T, C
        wembs = sm @ value
        return wembs