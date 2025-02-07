import torch
import torch.nn as nn

class ComplexConv2d(nn.Module):
    def __init__(self, conv2d: nn.Conv2d, imag_init_mean = 0, imag_init_std=1e-4):
        super(ComplexConv2d, self).__init__()
        self.conv2d_re = conv2d
        self.conv2d_re.weight = nn.Parameter(conv2d.weight.clone())
        self.conv2d_im = nn.Conv2d(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=conv2d.bias is not None
        )
        # 虚部权重初始化为0附近的小随机数
        nn.init.normal_(self.conv2d_im.weight, mean=imag_init_mean, std=imag_init_std)
        if self.conv2d_im.bias is not None:
            nn.init.zeros_(self.conv2d_im.bias)

    def forward(self, in_re, in_im, output_type='complex'):
        # 计算实部和虚部的卷积
        out_re = self.conv2d_re(in_re) - self.conv2d_im(in_im)
        out_im = self.conv2d_re(in_im) + self.conv2d_im(in_re)

        if output_type == 'complex':
            return out_re, out_im
        elif output_type == 'projection':
            return out_re
        elif output_type == 'signed_norm':
            norm = torch.sqrt(out_re**2 + out_im**2)
            sign = torch.sign(out_re)
            return norm * sign
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
        
class ComplexLinear(nn.Module):
    def __init__(self, linear: nn.Linear, as_lora=False, r=None):
        super(ComplexLinear, self).__init__()
        self.linear_re = linear  # 实部权重
        self.linear_im = nn.Linear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None
        )
        # 虚部权重初始化为0附近的小随机数
        nn.init.normal_(self.linear_im.weight, mean=0, std=0.02)
        if self.linear_im.bias is not None:
            nn.init.zeros_(self.linear_im.bias)

        self.as_lora = as_lora
        if as_lora:
            assert r is not None, "r must be specified when as_lora is True"
            self.r = r
            # LoRA for real part
            self.lora_a_re = nn.Linear(linear.in_features, r, bias=False)
            self.lora_b_re = nn.Linear(r, linear.out_features, bias=False)
            # LoRA for imaginary part
            self.lora_a_im = nn.Linear(linear.in_features, r, bias=False)
            self.lora_b_im = nn.Linear(r, linear.out_features, bias=False)
            # Freeze original weights
            for param in self.linear_re.parameters():
                param.requires_grad = False
            for param in self.linear_im.parameters():
                param.requires_grad = False

    def forward(self, in_re, in_im, output_type='complex'):
        if self.as_lora:
            # LoRA for real part
            deltaW_re = self.lora_b_re(self.lora_a_re(in_re))
            # LoRA for imaginary part
            deltaW_im = self.lora_b_im(self.lora_a_im(in_im))
            # Apply LoRA
            out_re = self.linear_re(in_re) + deltaW_re
            out_im = self.linear_im(in_im) + deltaW_im
        else:
            # Standard complex linear transformation
            out_re = self.linear_re(in_re) - self.linear_im(in_im)
            out_im = self.linear_re(in_im) + self.linear_im(in_re)

        if output_type == 'complex':
            return out_re, out_im
        elif output_type == 'projection':
            return out_re
        elif output_type == 'signed_norm':
            norm = torch.sqrt(out_re**2 + out_im**2)
            sign = torch.sign(out_re)
            return norm * sign
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")
        
if __name__ == "__main__":  
    # 示例：将普通的 Conv2d 和 Linear 转换为 ComplexConv2d 和 ComplexLinear
    conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    complex_conv2d = ComplexConv2d(conv2d)

    linear = nn.Linear(in_features=128, out_features=64)
    complex_linear = ComplexLinear(linear, as_lora=True, r=8)

    # 输入张量
    in_re = torch.randn(1, 3, 32, 32)
    in_im = torch.randn(1, 3, 32, 32)

    # 前向传播
    out_re, out_im = complex_conv2d(in_re, in_im, output_type='complex')
    print(out_re.shape, out_im.shape)

    in_re = torch.randn(1, 128)
    in_im = torch.randn(1, 128)
    # 通过复杂线性层
    out_re, out_im = complex_linear(in_re, in_im, output_type='complex')
    print(out_re.shape, out_im.shape)

    out_proj = complex_linear(in_re, in_im, output_type='projection')
    print(out_proj.shape)

    out_signed_norm = complex_linear(in_re, in_im, output_type='signed_norm')
    print(out_signed_norm.shape)