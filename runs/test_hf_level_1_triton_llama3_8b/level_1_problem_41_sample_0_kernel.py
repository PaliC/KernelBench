batch_size = 16
features = 64
sequence_length = 128

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return []