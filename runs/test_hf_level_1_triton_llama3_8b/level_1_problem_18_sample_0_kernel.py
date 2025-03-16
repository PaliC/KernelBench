M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(K, M).cuda()
    B = torch.randn(N, K).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed