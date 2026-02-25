from . import (
    pointwise,
    pairwise,
    listwise,
)


SAMPLER_REGISTRY = {
    "pointwise": pointwise.NegativeSampler,
    "pairwise": pairwise.NegativeSampler,
    "listwise": listwise.NegativeSampler,
    "msr": pointwise.NegativeSampler,
}