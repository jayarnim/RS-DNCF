from torch.nn.utils.rnn import pad_sequence


def histories_generator(selector):
    # select hist per anchor
    kwargs = dict()
    hist_indices = selector(**kwargs)

    # padding
    kwargs = dict(
        sequences=hist_indices, 
        batch_first=True, 
        padding_value=0,
    )
    hist_indices_padded = pad_sequence(**kwargs)

    return hist_indices_padded