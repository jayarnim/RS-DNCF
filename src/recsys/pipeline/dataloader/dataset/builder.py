from .dataset.registry import DATASET_REGISTRY


def dataset_builder(df, sampler, strategy, schema):
    cls = DATASET_REGISTRY[strategy]
    kwargs = dict(
        df=df,
        sampler=sampler,
        schema=schema,
    )
    return cls(**kwargs)
