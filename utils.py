import torch


def checkpoint(path, model, optimizer, loss, **kwargs):

    params = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        loss=loss
    )
    params.update(kwargs)
    
    torch.save(params, path)


def visualize(outputs, true_values, cfa=None):
    pass
