import torch 

lr = 0.01
epochs = 10

def deviceOption():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    
    return device
    