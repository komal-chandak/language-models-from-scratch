from .base import Tokenizer, get_stats, merge
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size>=256
        num_merges = vocab_size-256

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {i:bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                stats = get_stats(chunk_ids, stats)

            top_pair = max(stats, key=stats.get)
            idx = 256+i
            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]  # merge process for each chunk

            merges[top_pair]=idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({vocab[idx]}) had {stats[top_pair]} occurrences")

            self.merges = merges
            self.vocab = vocab

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids)>=2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids,pair,idx)
        return ids

    def encode_ordinary(self, text):
        """encoding without any special token"""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special = "none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens

        elif allowed_special == "none":
            special = None
        
        elif allowed_special == "none_raise":
            special = None
            assert all(tokens not in text for tokens in self.special_tokens)
        
        elif isinstance(allowed_special, set):
            special = {k:v for k, v in self.special_tokens() if k in allowed_special}
        
        else:
            ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids=[]
        for chunk in special_chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}