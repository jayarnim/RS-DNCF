import random


class NegativeSampler(object):
    def __init__(
        self,
        candidates: dict, 
        ratio: int,
        seed: int,
    ):
        super().__init__()
        self.candidates = candidates
        self.ratio = ratio
        self.rng = random.Random(seed)

    def __call__(self, user):
        return self.rng.choice(self.candidates[user])