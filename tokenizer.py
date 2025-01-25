import tiktoken
from typing import List


class Tokenizer:
    def __init__(self, tokenizer_model=None):
        # Load the cl100k_base tokenizer
        self.enc = tiktoken.get_encoding("cl100k_base")

        # Define special tokens
        # Use EOT as BOS (beginning of sequence)
        self.bos_id = self.enc.eot_token
        self.eos_id = self.enc.eot_token  # Use EOT as EOS (end of sequence)
        self.pad_id = self.enc.eot_token  # Use EOT as PAD (padding token)

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        # Encode the input string
        t = self.enc.encode(s)

        # Add BOS and EOS tokens if specified
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        # Decode the list of token IDs into a string
        return self.enc.decode(t)
