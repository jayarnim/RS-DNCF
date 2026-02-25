from .sampler.registry import SAMPLER_REGISTRY


def sampler_builder(candidates, strategy, ratio, seed):
    kwargs = dict(
        candidates=candidates,
        ratio=ratio,
        seed=seed,
    )
    cls = SAMPLER_REGISTRY[strategy]
    return cls(**kwargs)