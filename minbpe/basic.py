from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)   # list of integer (0 to 255)

        # iterative merge process
        tokens = list(ids)   # a copy
        vocab = {i:bytes([i]) for i in range(256)}
        merges = {}
        for i in range(num_merges):
            stats = get_stats(tokens)                  # get the count for each pair
            top_pair = max(stats, key=stats.get)    # select the most occuring pair
            idx = 256 + i                           # mint a new token
            tokens = merge(tokens, top_pair, idx)   # replace all the occurences of the pair with the new token
            merges[top_pair] = idx                  # save the merge
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]    # add to the vocab

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurrences")
        
        # to be used in encoder, decoder
        self.merges = merges
        self.vocab = vocab


    def encode(self, text):
        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)   # list of integer (0 to 255)

        while len(ids)>=2:
            stats = get_stats(ids)   # get the count for each pair
            pair = min(stats, key = lambda p: self.merges.get(p, float('inf')))   # find the pair as per the merging order
            if pair in self.merges:
                idx = self.merges[pair]
                ids = merge(ids,pair,idx)
            else:
                break
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
