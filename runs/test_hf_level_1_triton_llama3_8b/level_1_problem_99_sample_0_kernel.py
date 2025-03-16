def get_inputs():
    return [torch.randn(128, 4096), torch.randn(128, 4096), torch.randn(128, 4096)]

def get_init_inputs():
    return [1.0]  # Default margin