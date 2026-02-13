"""StateLanguage — defines vocabularies by quantizing continuous values into discrete token buckets."""

import json


class StateLanguage:
    """Defines the vocabulary for a state language."""

    def __init__(self, name, schema):
        """
        schema: dict mapping prefix to (min_val, max_val, num_buckets)
        Example: {'C': (0, 100, 10), 'M': (0, 100, 10), 'N': (0, 1, 2)}

        This creates tokens: C0 C1 C2...C9 M0 M1...M9 N0 N1
        Plus: BOS (beginning of sequence)
        """
        self.name = name
        self.schema = schema
        self.tokens = ['<BOS>']
        self.prefix_map = {}  # prefix -> list of token indices

        for prefix, (lo, hi, buckets) in schema.items():
            bucket_tokens = []
            for b in range(buckets):
                token = f"{prefix}{b}"
                bucket_tokens.append(len(self.tokens))
                self.tokens.append(token)
            self.prefix_map[prefix] = bucket_tokens

        self.vocab_size = len(self.tokens)
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for i, t in enumerate(self.tokens)}
        self.BOS = 0

    def encode_value(self, prefix, value):
        """Quantize a continuous value into a token id."""
        lo, hi, buckets = self.schema[prefix]
        bucket = int((value - lo) / (hi - lo) * buckets)
        bucket = max(0, min(buckets - 1, bucket))
        token = f"{prefix}{bucket}"
        return self.stoi[token]

    def decode_token(self, token_id):
        """Get the string representation of a token."""
        return self.itos.get(token_id, '?')

    def encode_observation(self, obs_dict):
        """Encode a dict of {prefix: value} into a token sequence."""
        tokens = [self.BOS]
        for prefix in self.schema:  # deterministic order
            if prefix in obs_dict:
                tokens.append(self.encode_value(prefix, obs_dict[prefix]))
        return tokens

    def decode_sequence(self, token_ids):
        """Decode token ids back to readable form."""
        return ' '.join(self.decode_token(t) for t in token_ids)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'name': self.name, 'schema': self.schema}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d['name'], {k: tuple(v) for k, v in d['schema'].items()})
