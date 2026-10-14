"""Editable submission. Engine owns a request's KV cache; no cross-request reuse."""
class Engine:
    def __init__(self, model):
        self.model = model
        self.cache = None

    def prefill(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        self.cache = out.past_key_values
        return out.logits[:, -1, :]

    def decode(self, token):
        out = self.model(input_ids=token, past_key_values=self.cache,
                         use_cache=True, logits_to_keep=1)
        self.cache = out.past_key_values
        return out.logits[:, -1, :]
