import pandas as pd
from .dataloader import dataloader_generator


def dataloader_builder(
    split: dict[str, pd.DataFrame], 
    candidates: dict, 
    cfg,
):
    dataloader = dict()
    converter = dict(
        trn="opt",
        val="msr",
        tst="msr",
    )

    for split_type, split_vals in split.items():
        TASK = converter[split_type]
        STRATEGY = (
            cfg.strategy
            if TASK=="opt"
            else "msr"
        )
        RATIO = cfg.ratio[TASK]

        kwargs = dict(
            df=split_vals,
            candidates=candidates,
            schema=cfg.schema,
            strategy=STRATEGY,
            ratio=RATIO,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
        dataloader[split_type] = dataloader_generator(**kwargs)

    return dataloader