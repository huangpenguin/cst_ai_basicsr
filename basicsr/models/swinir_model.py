import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY

from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):
    def _forward_net(self, net: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
        return net(img)

    def _test_full(self, net: torch.nn.Module, window_size: int, scale: int) -> torch.Tensor:
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        output = self._forward_net(net, img)
        _, _, out_h, out_w = output.size()
        return output[:, :, 0 : out_h - mod_pad_h * scale, 0 : out_w - mod_pad_w * scale]

    def _test_tile(
        self,
        net: torch.nn.Module,
        *,
        tile_size: int,
        tile_overlap: int,
        window_size: int,
        scale: int,
    ) -> torch.Tensor:
        _, _, h, w = self.lq.size()
        tile = min(tile_size, h, w)
        if tile % window_size != 0:
            tile = tile - (tile % window_size)
            if tile <= 0:
                tile = window_size
        stride = max(tile - tile_overlap, window_size)
        h_idx_list = list(range(0, max(h - tile, 0) + 1, stride))
        w_idx_list = list(range(0, max(w - tile, 0) + 1, stride))
        if not h_idx_list or h_idx_list[-1] != h - tile:
            h_idx_list.append(max(h - tile, 0))
        if not w_idx_list or w_idx_list[-1] != w - tile:
            w_idx_list.append(max(w - tile, 0))

        output = torch.zeros((self.lq.size(0), self.lq.size(1), h * scale, w * scale), device=self.lq.device)
        weight = torch.zeros_like(output)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = self.lq[:, :, h_idx : h_idx + tile, w_idx : w_idx + tile]
                mod_pad_h, mod_pad_w = 0, 0
                _, _, patch_h, patch_w = in_patch.size()
                if patch_h % window_size != 0:
                    mod_pad_h = window_size - patch_h % window_size
                if patch_w % window_size != 0:
                    mod_pad_w = window_size - patch_w % window_size
                in_patch = F.pad(in_patch, (0, mod_pad_w, 0, mod_pad_h), "reflect")
                out_patch = self._forward_net(net, in_patch)
                _, _, out_h, out_w = out_patch.size()
                out_patch = out_patch[
                    :,
                    :,
                    0 : out_h - mod_pad_h * scale,
                    0 : out_w - mod_pad_w * scale,
                ]
                out_h, out_w = out_patch.size(-2), out_patch.size(-1)
                output[:, :, h_idx * scale : h_idx * scale + out_h, w_idx * scale : w_idx * scale + out_w].add_(
                    out_patch
                )
                weight[:, :, h_idx * scale : h_idx * scale + out_h, w_idx * scale : w_idx * scale + out_w].add_(1.0)
        return output.div_(weight)

    def test(self):
        window_size = self.opt["network_g"]["window_size"]
        scale = self.opt.get("scale", 1)
        val_opt = self.opt.get("val") or {}
        tile_size = val_opt.get("tile_size")
        tile_overlap = int(val_opt.get("tile_overlap", 32))

        if hasattr(self, "net_g_ema"):
            net = self.net_g_ema
            net.eval()
            with torch.no_grad():
                if tile_size:
                    self.output = self._test_tile(
                        net,
                        tile_size=int(tile_size),
                        tile_overlap=tile_overlap,
                        window_size=window_size,
                        scale=scale,
                    )
                else:
                    self.output = self._test_full(net, window_size, scale)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if tile_size:
                    self.output = self._test_tile(
                        self.net_g,
                        tile_size=int(tile_size),
                        tile_overlap=tile_overlap,
                        window_size=window_size,
                        scale=scale,
                    )
                else:
                    self.output = self._test_full(self.net_g, window_size, scale)
            self.net_g.train()
