from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralModulationAdapter(nn.Module):
    def __init__(self, channels: int, hidden: int = None, factor: float = 0.1):
        super().__init__()
        hidden = hidden or max(8, channels // 4)
        self.factor = factor
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x + self.factor * x * (self.net(x) - 0.5)


class SpatialModulationAdapter(nn.Module):
    def __init__(self, channels: int, hidden: int = None, factor: float = 0.1):
        super().__init__()
        hidden = hidden or max(8, channels // 4)
        self.factor = factor
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x + self.factor * x * (self.net(x) - 0.5)


class FrequencyModulationAdapter(nn.Module):
    def __init__(self, channels: int, hidden: int = None, factor: float = 0.1):
        super().__init__()
        hidden = hidden or max(8, channels // 4)
        self.factor = factor
        self.down = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.up = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = torch.fft.rfft2(self.down(x), dim=(2, 3), norm="backward")
        amp = torch.abs(y)
        phase = torch.angle(y)
        amp = amp * torch.sigmoid(self.dw(amp))
        real = amp * torch.cos(phase)
        imag = amp * torch.sin(phase)
        y = torch.complex(real, imag)
        y = torch.fft.irfft2(y, s=(h, w), dim=(2, 3), norm="backward")
        return x + self.factor * self.up(F.gelu(y))


def _zero_module(module: nn.Module):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.zeros_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)
    return module


class SpectralFeatureModulation(nn.Module):
    """Channel/spectral modulation branch for HSI feature adapters."""

    def __init__(self, channels: int, hidden: int = None, zero_init: bool = True):
        super().__init__()
        hidden = hidden or max(8, channels // 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        if zero_init:
            _zero_module(self.net[-1])

    def forward(self, x):
        return x * torch.tanh(self.net(x))


class SpatialFeatureModulation(nn.Module):
    """SMA branch following Adapt-ICMH's bottleneck spatial modulation."""

    def __init__(self, channels: int, hidden: int = None, zero_init: bool = True):
        super().__init__()
        hidden = hidden or max(8, channels // 4)
        self.down1 = nn.Conv2d(channels, hidden, 1)
        self.down2 = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden)
        self.up = nn.Conv2d(hidden, channels, 1)
        if zero_init:
            _zero_module(self.up)

    def forward(self, x):
        base = self.down1(x)
        gate = self.dw(self.down2(x))
        return self.up(F.gelu(base * gate))


class FrequencyFeatureModulation(nn.Module):
    """FMA branch that modulates FFT amplitude while preserving phase."""

    def __init__(self, channels: int, hidden: int = None, zero_init: bool = True):
        super().__init__()
        hidden = hidden or max(8, channels // 4)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.inter = nn.Conv2d(hidden, hidden, 1)
        self.up = nn.Conv2d(hidden, channels, 1)
        if zero_init:
            _zero_module(self.up)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = torch.fft.rfft2(self.down(x), dim=(2, 3), norm="backward")
        amp = torch.abs(y)
        phase = torch.angle(y)
        amp_mod = self.inter(F.gelu(self.dw(amp)))
        amp = amp * torch.sigmoid(amp_mod)
        y = torch.complex(amp * torch.cos(phase), amp * torch.sin(phase))
        y = torch.fft.irfft2(y, s=(h, w), dim=(2, 3), norm="backward")
        return self.up(F.gelu(y))


class HSIFeatureAdapter(nn.Module):
    """Parallel SpMA/SMA/FMA feature adapter for multi-scale codec features."""

    def __init__(
        self,
        channels: int,
        axes: Iterable[str] = ("spectral", "spatial", "frequency"),
        hidden: int = None,
        factor: float = 1.0,
        zero_init: bool = True,
    ):
        super().__init__()
        axes = tuple(axes)
        modules = []
        for axis in axes:
            if axis in ("spectral", "spma", "channel"):
                modules.append(SpectralFeatureModulation(channels, hidden, zero_init=zero_init))
            elif axis in ("spatial", "sma"):
                modules.append(SpatialFeatureModulation(channels, hidden, zero_init=zero_init))
            elif axis in ("frequency", "fma"):
                modules.append(FrequencyFeatureModulation(channels, hidden, zero_init=zero_init))
            else:
                raise ValueError(f"Unknown feature adapter axis={axis}")
        self.axes = axes
        self.factor = float(factor)
        self.branches = nn.ModuleList(modules)

    def forward(self, x):
        if not self.branches:
            return x
        residual = self.branches[0](x)
        for branch in self.branches[1:]:
            residual = residual + branch(x)
        return x + self.factor * residual


class HSITaskAdapter(nn.Module):
    """SpMA/SMA/FMA-style latent adapter for HSI compression latents."""

    def __init__(
        self,
        channels: int,
        axes: Iterable[str] = ("spectral", "spatial", "frequency"),
        hidden: int = None,
        factor: float = 0.1,
    ):
        super().__init__()
        axes = tuple(axes)
        modules = []
        for axis in axes:
            if axis in ("spectral", "spma"):
                modules.append(SpectralModulationAdapter(channels, hidden, factor))
            elif axis in ("spatial", "sma"):
                modules.append(SpatialModulationAdapter(channels, hidden, factor))
            elif axis in ("frequency", "fma"):
                modules.append(FrequencyModulationAdapter(channels, hidden, factor))
            else:
                raise ValueError(f"Unknown adapter axis={axis}")
        self.axes = axes
        self.adapters = nn.ModuleList(modules)

    def forward(self, x, task_id=None):
        del task_id
        for adapter in self.adapters:
            x = adapter(x)
        return x


def make_hsi_feature_adapter(
    channels: int,
    axes: Tuple[str, ...] = ("spectral", "spatial", "frequency"),
    adapter_factor: float = 1.0,
    hidden: int = None,
    zero_init: bool = True,
):
    return HSIFeatureAdapter(
        channels,
        axes=axes,
        hidden=hidden,
        factor=adapter_factor,
        zero_init=zero_init,
    )
