from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


# RRDBNet architecture adapted from ai-forever/Real-ESRGAN, BSD 3-Clause.
@dataclass(frozen=True)
class RealESRGANConfig:
    tile_size: int = 320
    tile_pad: int = 24
    batch_size: int = 8
    scale: int = 2


class RealESRGAN2x:
    """Minimal Real-ESRGAN x2 inference wrapper for ai-forever weights."""

    def __init__(self, weights_path, device: str, dtype, config: RealESRGANConfig | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or RealESRGANConfig()
        self.model = _build_rrdb_net(num_in_ch=3, num_out_ch=3, scale=self.config.scale)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device, dtype=self.dtype)

    def run(self, image: Image.Image, config: RealESRGANConfig | None = None) -> Image.Image:
        config = config or self.config
        source = np.asarray(image.convert("RGB"))
        padded = _pad_to_tile(source, config.tile_size)
        tiles, layout = _split_tiles(padded, config.tile_size, config.tile_pad)
        outputs = []
        with self.torch.inference_mode():
            for index in range(0, len(tiles), config.batch_size):
                batch_tiles = tiles[index : index + config.batch_size]
                batch = self.torch.cat(
                    [_image_to_tensor(tile, self.torch, self.device, self.dtype) for tile in batch_tiles],
                    dim=0,
                )
                output = self.model(batch).clamp_(0, 1)
                outputs.extend(_tensor_to_image(output[tile_index : tile_index + 1]) for tile_index in range(output.shape[0]))
        merged = _merge_tiles(outputs, layout, config.tile_size, config.tile_pad, config.scale)
        target_height = image.height * config.scale
        target_width = image.width * config.scale
        result = merged[:target_height, :target_width, :]
        return Image.fromarray(result, mode="RGB")


def _pad_to_tile(image: np.ndarray, tile_size: int) -> np.ndarray:
    height, width, _ = image.shape
    pad_height = (tile_size - height % tile_size) % tile_size
    pad_width = (tile_size - width % tile_size) % tile_size
    if pad_height == 0 and pad_width == 0:
        return image
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def _split_tiles(image: np.ndarray, tile_size: int, tile_pad: int) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    height, width, _ = image.shape
    rows = height // tile_size
    cols = width // tile_size
    padded = np.pad(image, ((tile_pad, tile_pad), (tile_pad, tile_pad), (0, 0)), mode="reflect")
    tiles = []
    for row in range(rows):
        for col in range(cols):
            top = row * tile_size
            left = col * tile_size
            tiles.append(padded[top : top + tile_size + tile_pad * 2, left : left + tile_size + tile_pad * 2, :])
    return tiles, (rows, cols, height, width)


def _merge_tiles(
    tiles: list[np.ndarray],
    layout: tuple[int, int, int, int],
    tile_size: int,
    tile_pad: int,
    scale: int,
) -> np.ndarray:
    rows, cols, height, width = layout
    scaled_tile = tile_size * scale
    scaled_pad = tile_pad * scale
    merged = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
    tile_iter = iter(tiles)
    for row in range(rows):
        for col in range(cols):
            tile = next(tile_iter)
            tile = tile[scaled_pad : scaled_pad + scaled_tile, scaled_pad : scaled_pad + scaled_tile, :]
            top = row * scaled_tile
            left = col * scaled_tile
            merged[top : top + scaled_tile, left : left + scaled_tile, :] = tile
    return merged


def _image_to_tensor(image: np.ndarray, torch, device, dtype):
    array = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).to(device=device, dtype=dtype)


def _tensor_to_image(tensor) -> np.ndarray:
    array = tensor.squeeze(0).detach().float().cpu().numpy()
    array = np.transpose(array, (1, 2, 0)) * 255.0
    return np.clip(array.round(), 0, 255).astype(np.uint8)


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    import torch
    from torch import nn
    from torch.nn import init
    from torch.nn.modules.batchnorm import _BatchNorm

    if not isinstance(module_list, list):
        module_list = [module_list]
    with torch.no_grad():
        for module in module_list:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    init.kaiming_normal_(layer.weight, **kwargs)
                    layer.weight.data *= scale
                    if layer.bias is not None:
                        layer.bias.data.fill_(bias_fill)
                elif isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, **kwargs)
                    layer.weight.data *= scale
                    if layer.bias is not None:
                        layer.bias.data.fill_(bias_fill)
                elif isinstance(layer, _BatchNorm):
                    init.constant_(layer.weight, 1)
                    if layer.bias is not None:
                        layer.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwargs):
    from torch import nn

    return nn.Sequential(*(basic_block(**kwargs) for _ in range(num_basic_block)))


def pixel_unshuffle(x, scale: int):
    batch, channels, height, width = x.size()
    out_channel = channels * (scale**2)
    view = x.view(batch, channels, height // scale, scale, width // scale, scale)
    return view.permute(0, 1, 3, 5, 2, 4).reshape(batch, out_channel, height // scale, width // scale)


def _build_rrdb_net(num_in_ch: int, num_out_ch: int, scale: int):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class _ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class _RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class _RRDBNet(nn.Module):
        def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            if scale == 2:
                num_in_ch *= 4
            elif scale == 1:
                num_in_ch *= 16
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = make_layer(_RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = pixel_unshuffle(x, scale=2) if self.scale == 2 else x
            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return _RRDBNet(num_in_ch=num_in_ch, num_out_ch=num_out_ch, scale=scale)
