import torch
import random
import torch.nn.functional as F
import torch.nn as nn

# hyperparameters

# scaled network: to be ran on a gpu (ran using google colab: T4 ~ 52 mins )
batch_size = 64  # how many independent sequence will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout= 0.2
# ------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print(f'len of input text: {len(text)}')
# print(text[:400])

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]   # encoder: take a string and output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder = take a list of integers, output a string

# print(encode("hello"))
# print(decode(encode('hi')))

# encode the entire text dataset
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.dtype, data.shape)
# print(data[:100])

# training and val splits: 90% training, rest val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# print(len(train_data), len(val_data))

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix =  torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y= torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  
        self.value = nn.Linear(n_embd, head_size, bias=False)  
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        v = self.value(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores "affinities"
        # scaling by C because n_embd or C is the head_size here
        wei = q @ k.transpose(-2,-1) * C **-0.5  # (B, T, C) @ (B, C, T) --> (B, T, T)   
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)
        # perform weighted aggregation of values
        out = wei @ v   # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block : communication followed by compuatation"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # slight deivation from og paper:pre norm formulation - the layer norm is applied before the sa and ffwd  
        x = x + self.sa(self.ln1(x))  # residual connections
        x = x + self.ffwd(self.ln2(x)) # residual connections
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        # self.sa_head = Head(n_embd)  # single head

        # multiple head
        # self.sa_head = MultiHeadAttention(4, n_embd//4)  # i.e. 4 heads of 8 dimensional self attention
        # self.ffwd = FeedForward(n_embd)
        
        # multiple head as a block - conversation(attention layer: sa) and computation layer (feedfoward: ffwd) interspersed and repeated sequentially
        # self.blocks = nn.Sequential(Block(n_embd, n_head=4),
        #                             Block(n_embd, n_head=4),
        #                             Block(n_embd, n_head=4),
        #                             nn.LayerNorm(n_embd),
        #                             )
        
        # more cleaner version:
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self,idx,targets=None):
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # ( B, T, C) arranged as batch, time, channel(n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb # (B, T, C)   # token identities + the position at which the token occurs
        
        # x = self.sa_head(x) # apply one or multiple head (as per the init) of self attention (B, T, C)
        # x = self.ffwd(x) # applied to each token individually (B, T, C)
        
        # with block
        x = self.blocks(x)
        logits = self.lm_head(x)  #(B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)  # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_tokens):
            # crop the idx to the last block_size tokens because we have positional embeddings now and this will run of scope otherwise
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)        
            # focus only on the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            # append sampled idx to running sequence
            idx = torch.cat((idx,idx_next), dim=1) # (B, T+1)
        return idx

model = TransformerLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr = 1e-3)

# training loop
for iter in range(max_iters):

    # every once in a while eval the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss  = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context,max_tokens=400)[0].tolist()))