import torch


# if no type is supplied, try gpu if avaiable, user cpu otherwise
def getDevice(type=''):
    checkGPU = torch.cuda.is_available()

    assert type != 'cuda' or checkGPU,\
        "cuda requested but not avaiable"

    if type == '':
        return 'cuda' if checkGPU else 'cpu'
    return type
