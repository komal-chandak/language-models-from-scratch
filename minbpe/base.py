import unicodedata

def get_stats(ids, counts=None): 
    """counts: to keep on adding to the existing count in case of regex splits"""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(tokens, pair, idx):
    new_ids = []
    i=0
    while i<len(tokens):
        if  i<len(tokens)-1 and pair[0]==tokens[i] and pair[1]==tokens[i+1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(tokens[i])
            i+=1
    return new_ids

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(token: bytes) -> str:
    s = token.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s

class Tokenizer:
    """Base class for Tokenizers"""
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab_()

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError
    
    def _build_vocab_(self):
        vocab = {i:bytes([i]) for i in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix):

        # write the model: to be used in load() later
        model_file = file_prefix + '.model'
        with open(model_file, 'w') as f:

            # write the version, pattern, special tokens and their len, merges in the model file
            f.write("minbpe v1\n")   
            f.write(f"{self.pattern}\n")
            
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges:
                # saving only keys of merges i.e. pairs not the ranks
                f.write(f"{idx1} {idx2}\n")

        # write the vocab: for human inspection only
        vocab_file = file_prefix + '.vocab'
        inverted_merges = {idx:pair for pair,idx in self.merges.items()}
        
        # note: many tokens may be partial utf-8 sequences and cannot be decoded into valid strings.
        # Here we're using errors='replace' to replace them with the replacement char �.
        # this also means that we couldn't possibly use .vocab in load() because decoding in this way is a lossy operation!

        with open(vocab_file, 'w', encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)  # convert to str

                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith('.model')
        special_tokens = {}
        merges = {}
        idx = 256
        # read
        with open(model_file, 'r', encoding="utf-8") as f:

            version = f.readline().strip()
            assert version == 'minbpe v1'

            self.pattern = f.readline().strip()

            num_special = int(f.readline().strip())
            
            for _ in range(num_special):
                special,special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            
            for line in f:
                idx1, idx2 = map(int, line.strip().split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab_()
