import torch
from torch import nn

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图









#######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义h_sigmoid激活函数，这是一种硬Sigmoid函数
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)  # 使用ReLU6实现

    def forward(self, x):
        return self.relu(x + 3) / 6  # 公式为ReLU6(x+3)/6，模拟Sigmoid激活函数

# 定义h_swish激活函数，这是基于h_sigmoid的Swish函数变体
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)  # 使用上面定义的h_sigmoid

    def forward(self, x):
        return x * self.sigmoid(x)  # 公式为x * h_sigmoid(x)

# 定义Coordinate Attention模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 定义水平和垂直方向的自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向

        mip = max(8, inp // reduction)  # 计算中间层的通道数

        # 1x1卷积用于降维
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化
        self.act = h_swish()  # 激活函数

        # 两个1x1卷积，分别对应水平和垂直方向
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x  # 保存输入作为残差连接

        n, c, h, w = x.size()  # 获取输入的尺寸
        x_h = self.pool_h(x)  # 水平方向池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 垂直方向池化并交换维度以适应拼接

        y = torch.cat([x_h, x_w], dim=2)  # 拼接水平和垂直方向的特征
        y = self.conv1(y)  # 通过1x1卷积降维
        y = self.bn1(y)  # 批归一化
        y = self.act(y)  # 激活函数

        x_h, x_w = torch.split(y, [h, w], dim=2)  # 将特征拆分回水平和垂直方向
        x_w = x_w.permute(0, 1, 3, 2)  # 恢复x_w的原始维度

        a_h = self.conv_h(x_h).sigmoid()  # 通过1x1卷积并应用Sigmoid获取水平方向的注意力权重
        a_w = self.conv_w(x_w).sigmoid()  # 通过1x1卷积并应用Sigmoid获取垂直方向的注意力权重

        out = identity * a_w * a_h  # 应用注意力权重到输入特征，并与残差连接相乘

        return out  # 返回输出


#############################################################
import torch
import torch.nn.functional as F
import math

"""
1.内存效率提升：原始实现需要扩展所有中间变量来执行不同的激活函数，而此代码中将计算重新制定为使用不同的基函数激活输入，
  然后线性组合它们。这种重新制定可以显著降低内存成本，并将计算变得更加高效。

2.正则化方法的改变：原始实现中使用的L1正则化需要对张量进行非线性操作，与重新制定的计算不兼容。
  因此，此代码中将L1正则化改为对权重的L1正则化，这更符合神经网络中常见的正则化方法，并且与重新制定的计算兼容。

3.激活函数缩放选项：原始实现中包括了每个激活函数的可学习缩放，但这个库提供了一个选项来禁用这个特性。
  禁用缩放可以使模型更加高效，但可能会影响结果。

4.参数初始化的改变：为了解决在MNIST数据集上的性能问题，此代码修改了参数的初始化方式，使用kaiming初始化。
"""

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 网格大小，默认为 5
        spline_order=3, # 分段多项式的阶数，默认为 3
        scale_noise=0.1,  # 缩放噪声，默认为 0.1
        scale_base=1.0,   # 基础缩放，默认为 1.0
        scale_spline=1.0,    # 分段多项式的缩放，默认为 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
        grid_eps=0.02,
        grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = ( # 生成网格
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # 初始化基础权重和分段多项式权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1 # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module): # 封装了一个KAN神经网络模型，可以用于对数据进行拟合和预测。
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        初始化 KAN 模型。

        参数:
            layers_hidden (list): 包含每个隐藏层输入特征数量的列表。
            grid_size (int): 网格大小，默认为 5。
            spline_order (int): 分段多项式的阶数，默认为 3。
            scale_noise (float): 缩放噪声，默认为 0.1。
            scale_base (float): 基础缩放，默认为 1.0。
            scale_spline (float): 分段多项式的缩放，默认为 1.0。
            base_activation (torch.nn.Module): 基础激活函数，默认为 SiLU。
            grid_eps (float): 网格调整参数，默认为 0.02。
            grid_range (list): 网格范围，默认为 [-1, 1]。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False): # 调用每个KANLinear层的forward方法，对输入数据进行前向传播计算输出。
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否更新网格。默认为 False。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):#计算正则化损失的方法，用于约束模型的参数，防止过拟合。
        """
        计算正则化损失。

        参数:
            regularize_activation (float): 正则化激活项的权重，默认为 1.0。
            regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
            torch.Tensor: 正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
