import pandas as pd
from typing import Literal
from torch.utils.data import DataLoader
from .collate.builder import collate_fn_builder
from .sampler.builder import sampler_builder
from .dataset.builder import dataset_builder


def dataloader_generator(
    df: pd.DataFrame,
    candidates: dict,
    schema,
    strategy: Literal["pointwise", "pairwise", "listwise", "msr"], 
    ratio: int, 
    batch_size: int, 
    shuffle: bool, 
    seed: int,
):
    kwargs = dict(
        candidates=candidates,
        strategy=strategy,
        ratio=ratio,
        seed=seed,
    )
    sampler = sampler_builder(**kwargs)

    kwargs = dict(
        df=df,
        sampler=sampler,
        strategy=strategy,
        schema=schema,
    )
    dataset = dataset_builder(**kwargs)

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_builder(strategy),
    )
    dataloader = DataLoader(**kwargs)

    return dataloader