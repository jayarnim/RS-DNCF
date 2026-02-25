from ..config.model import (
    DGMFCfg,
    DMLPCfg,
    DNMFCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="dgmf":
        return dgmf(cfg)
    elif model=="dmlp":
        return dmlp(cfg)
    elif model=="dnmf":
        return dnmf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def dgmf(cfg):
    return DGMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        num_factors=cfg["model"]["num_factors"],
        combiner=cfg["model"]["combiner"],
    )


def dmlp(cfg):
    return DMLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        num_factors=cfg["model"]["num_factors"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def dnmf(cfg):
    dgmf = DGMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        num_factors=cfg["model"]["dgmf"]["num_factors"],
        combiner=cfg["model"]["dgmf"]["combiner"],
    )
    dmlp = DMLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        num_factors=cfg["model"]["dmlp"]["num_factors"],
        hidden_dim=cfg["model"]["dmlp"]["hidden_dim"],
        dropout=cfg["model"]["dmlp"]["dropout"],
    )
    return DNMFCfg(
        dgmf=dgmf,
        dmlp=dmlp,
    )

