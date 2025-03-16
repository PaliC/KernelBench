M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed