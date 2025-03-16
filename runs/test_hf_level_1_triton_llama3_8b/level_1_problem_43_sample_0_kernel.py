batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, dim1, dim2, dim3)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]