import torch
import torch.nn as nn
import torch.nn.functional as F

class NativeQDQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, scale=1.0, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, **kwargs)
        # Use float32 and int32 exclusively to ensure TensorRT compatibility
        self.register_buffer('scales', torch.full((out_channels,), float(scale), dtype=torch.float32))
        self.register_buffer('zero_points', torch.zeros(out_channels, dtype=torch.int32))

    # Inside NativeQDQConv2d and NativeQDQLinear in qdq_layers.py

    def forward(self, x):
        # The .int() call is the critical fix for the "found Char" error
        w_qdq = torch.fake_quantize_per_channel_affine(
            self.weight, 
            self.scales, 
            self.zero_points.int(), # <--- FIX HERE
            0, 
            -128, 
            127
        )
        return self._conv_forward(x, w_qdq, self.bias)

class NativeQDQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=1.0, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        self.register_buffer('scales', torch.full((out_features,), float(scale), dtype=torch.float32))
        self.register_buffer('zero_points', torch.zeros(out_features, dtype=torch.int32))

    # Inside NativeQDQConv2d and NativeQDQLinear in qdq_layers.py

    def forward(self, x):
        # The .int() call is the critical fix for the "found Char" error
        w_qdq = torch.fake_quantize_per_channel_affine(
            self.weight, 
            self.scales, 
            self.zero_points.int(), # <--- FIX HERE
            0, 
            -128, 
            127
        )
        return F.linear(x, w_qdq, self.bias)