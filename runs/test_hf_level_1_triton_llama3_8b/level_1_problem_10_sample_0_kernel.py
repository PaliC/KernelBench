N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = torch.randn(N, M, K).cuda()
    B = torch.randn(K, L).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed