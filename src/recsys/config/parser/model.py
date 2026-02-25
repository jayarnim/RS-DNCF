from ..config.model import DGMFCfg, DMLPCfg, DNMFCfg


def model(cfg):
    cls = cfg["model"]["name"]

    if cls=="dgmf":
        return dgmf(cfg)
    elif cls=="dmlp":
        return dmlp(cfg)
    elif cls=="dnmf":
        return dnmf(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def dgmf(cfg):
    return DGMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def dmlp(cfg):
    return DMLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def dnmf(cfg):
    dgmf = DGMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["dgmf"],
    )
    dmlp = DMLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["dmlp"],
    )
    return DNMFCfg(
        dgmf=dgmf,
        dmlp=dmlp,
    )