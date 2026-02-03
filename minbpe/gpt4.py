from .regex import RegexTokenizer
import tiktoken

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    # https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py#L97
    parts = [bytes([b]) for b in token]  
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])  # get the rank for the 2 consecutive tokens
            if rank is not None and (min_rank is None or rank < min_rank):  # first iteration(first 2 tokens) or min_rank can be updated(rank for the pair is lower than the rank from previous pairs) 
                min_idx = i
                min_rank = rank

        # break if min_rank None i.e. no pair found in merge dict OR min_rank >=max i.e. Do not apply merges that happened after this token was created
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None

        # Remove tokens at positions min_idx and min_idx+1, Replace them with one merged token, Leave everything else untouched.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
        
    return parts

def recover_merges(mergeable_ranks):
    merges = {}

    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes

        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)

        # get the official tokenizer and its merges
        enc = tiktoken.get_encoding('cl100k_base')
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)

        # construct the vocab from merges
        vocab = {i:bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        self.vocab = vocab

        # for some reason, GPT-4 does not use raw byte values 0–255 as token IDs for single-byte tokens; it first remaps (permutes) them to a different numbering
        # as the tokens corresponding to individual bytes are permuted in a different order therefore we have to deal with it here.
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v:k for k,v in self.byte_shuffle.items()}

        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    # pretrained tokenizer, training not intended
    def train(self):
        raise NotImplementedError
    
    def _encode_chunk(self, text_bytes):
        # Take raw UTF-8 bytes from text, Replace each byte with GPT-4’s internal byte token ID 
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        # Reconstruct byte stream from token IDs, Convert GPT-4 byte IDs back to real byte values
        # Decode UTF-8 normally
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors='replace')
        return text
    
    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")
    
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")