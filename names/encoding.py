import numpy as np
from normality import ascii_text

ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789 .,?'


def encode_fixed(text, max_len=250):
    text = ascii_text(text) or ''
    text = text.lower()[:max_len].rjust(max_len, '?')
    indexes = [ALPHABET.find(c) for c in text]
    return np.eye(len(ALPHABET))[indexes]


print(encode_fixed('test me now'))
