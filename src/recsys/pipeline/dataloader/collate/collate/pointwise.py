import torch


def collate_fn(batch):
    user_list, item_list, label_list = zip(*batch)
    
    user_tensor = torch.tensor(
        data=user_list, 
        dtype=torch.long,
    )
    item_tensor = torch.tensor(
        data=item_list, 
        dtype=torch.long,
    )
    label_tensor = torch.tensor(
        data=label_list, 
        dtype=torch.float32,
    )
    
    return user_tensor, item_tensor, label_tensor