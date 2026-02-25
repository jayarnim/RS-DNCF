import torch


def collate_fn(batch):
    user_list, pos_list, neg_list = zip(*batch)

    user_tensor = torch.tensor(
        data=user_list, 
        dtype=torch.long,
    )
    pos_tensor = torch.tensor(
        data=pos_list, 
        dtype=torch.long,
    )
    neg_tensor = torch.tensor(
        data=neg_list, 
        dtype=torch.long,
    )

    return user_tensor, pos_tensor, neg_tensor