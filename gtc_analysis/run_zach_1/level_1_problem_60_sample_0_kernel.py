import torch
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch.nn as nn
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor


@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl
    .constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 8192
    tmp0 = tl.load(in_out_ptr0 + x2, None)
    tmp1 = tl.load(in_ptr0 + x1, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x2, tmp2, None)


def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 5, 7), (195, 105, 35, 7, 1))
    assert_size_stride(primals_2, (64,), (1,))
    assert_size_stride(primals_3, (16, 3, 64, 64, 64), (786432, 262144, 
        4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 
            1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False,
            output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (16, 64, 62, 60, 60), (92256, 1440, 23, 
            1, 1))
        buf1 = buf0
        del buf0
        get_raw_stream(0)
        triton_poi_fused_convolution_0[grid(147456)](buf1, primals_2, 
            147456, XBLOCK=1024, num_warps=4, num_stages=1)
        del primals_2
    return reinterpret_tensor(buf1, (16, 62, 60, 60, 64), (92256, 1440, 
        23, 1, 1), 0
        ), primals_1, primals_3


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return self.conv3d(x)

    def forward_pass(self, x, epoch):
        y = self.forward(x)
        return y


def get_inputs():
    x = torch.randn(batch_size, in_channels, width, height, depth)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization


class Conv3D(nn.Module):
    """
    This class implements a 3D convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Constructor method for the 3D convolutional layer.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolutional kernel
        :param bias: boolean indicating whether to add bias  
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv3d = nn.Conv3d(in_channels=self.in_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)

    def forward(self, x):
        """
        Performs a forward pass of x through the 3D convolutional layer.
        :param x: input activation
        :return: output activation
        """
        x = self.conv3d(x)
        return x


class ResidualBlock(nn.Module):
    """
    This class implements a residual block for a 3D convolutional network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Constructor method for the residual block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolutional kernel
        :param bias: boolean indicating whether to add bias  
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv3d_1 = Conv3D(in_channels=self.in_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)
        self.conv3d_2 = Conv3D(in_channels=self.out_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)

    def forward(self, x):
        """
        Performs a forward pass of x through the residual block.
        :param x: input activation
        :return: output activation
        """
        y = self.conv3d_1(x)
        y = self.conv3d_2(y)
        y = y + x
        return y


class ResidualBlock(nn.Module):
    """
    This class implements a residual block for a 3D convolutional network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Constructor method for the residual block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolutional kernel
        :param bias: boolean indicating whether to add bias  
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv3d_1 = Conv3D(in_channels=self.in_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)
        self.conv3d_2 = Conv3D(in_channels=self.out_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)

    def forward(self, x):
        """
        Performs a forward pass of x through the residual block.
        :param x: input activation
        :return: output activation
        """
        y = self.conv3d_1(x)
        y = self.conv3d_2(y)
        y = y + x
        return y


class ResidualBlock(nn.Module):
    """
    This class implements a residual block for a 3D convolutional network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Constructor method for the residual block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolutional kernel
        :param bias: boolean indicating whether to add bias  
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv3d_1 = Conv3D(in_channels=self.in_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)
        self.conv3d_2 = Conv3D(in_channels=self.out_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)

    def forward(self, x):
        """
        Performs a forward pass of x through the residual block.
        :param x: input activation
        :return: output activation
        """
        y = self.conv3d_1(x)
        y = self.conv3d_2(y)
        y = y + x
        return y


class ResidualBlock(nn.Module):
    """
    This class implements a residual block for a 3D convolutional network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Constructor method for the residual block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolutional kernel
        :param bias: boolean indicating whether to add bias  
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv3d_1 = Conv3D(in_channels=self.in_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)
        self.conv3d_2 = Conv3D(in_channels=self.out_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)

    def forward(self, x):
        """
        Performs a forward pass of x through the residual block.
        :param x: input activation
        :return: output activation
        """
        y = self.conv3d_1(x)
        y = self.conv3d_2(y)
        y = y + x
        return y


class ResidualBlock(nn.Module):
    """
    This class implements a residual block for a 3D convolutional network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Constructor method for the residual block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolutional kernel
        :param bias: boolean indicating whether to add bias  
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv3d_1 = Conv3D(in_channels=self.in_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)
        self.conv3d_2 = Conv3D(in_channels=self.out_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)

    def forward(self, x):
        """
        Performs a forward pass of x through the residual block.
        :param x: input activation
        :return: output activation
        """
        y = self.conv3d_1(x)
        y = self.conv3d_2(y)
        y = y + x
        return y


class ResidualBlock(nn.Module):
    """
    This class implements a residual block for a 3D convolutional network.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        """
        Constructor method for the residual block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the convolutional kernel
        :param bias: boolean indicating whether to add bias  
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv3d_1 = Conv3D(in_channels=self.in_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)
        self.conv3d_2 = Conv3D(in_channels=self.out_channels, out_channels=
            self.out_channels, kernel_size=self.kernel_size, bias=self.bias)

    def forward(self, x):
        """
        Performs a forward pass of x through the residual block.
        :param x: input activation
        :return: output activation
        """
        y = self.conv3d_1(x)
        y = self.conv3d_2(y)
        y = y + x
        return y


class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        return self.conv3d(x)

    def forward_pass(self, x, epoch):
        y = self.forward(x)
        return y