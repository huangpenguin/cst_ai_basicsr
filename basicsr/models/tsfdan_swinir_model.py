"""TSFDAN SwinIR model with optional peft LoRA fine-tuning."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any

import torch
from torch import nn

from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

from .swinir_model import SwinIRModel

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:  # pragma: no cover - enforced by pyproject dependency
    LoraConfig = None  # type: ignore[misc, assignment]
    PeftModel = None  # type: ignore[misc, assignment]
    get_peft_model = None  # type: ignore[misc, assignment]


def _is_peft_model(net: nn.Module) -> bool:
    return PeftModel is not None and isinstance(net, PeftModel)


@MODEL_REGISTRY.register()
class TSFDANSwinIRModel(SwinIRModel):
    """SwinIRModel subclass: load TSFDAN base, then wrap peft LoRA before DDP."""

    def __init__(self, opt: dict[str, Any]) -> None:
        # Skip SRModel.__init__ network build; replicate with LoRA inserted before DDP.
        from basicsr.models.base_model import BaseModel

        BaseModel.__init__(self, opt)

        self.net_g = build_network(opt["network_g"])
        self.net_g = self.net_g.to(self.device)
        self.print_network(self.net_g)

        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(self.net_g, load_path, self.opt["path"].get("strict_load_g", True), param_key)

        self.lora_cfg = self.opt.get("lora") or {}
        self.lora_enabled = bool(self.lora_cfg.get("enable", False))
        if self.lora_enabled:
            self.net_g = self._wrap_lora(self.net_g)

        self.net_g = self.model_to_device(self.net_g)

        if self.is_train:
            self.init_training_settings()

    def _wrap_lora(self, net: nn.Module) -> nn.Module:
        if get_peft_model is None or LoraConfig is None:
            raise ImportError("peft is required for lora.enable=true; run `uv add peft`")
        logger = get_root_logger()
        target_modules = self.lora_cfg.get(
            "target_modules",
            ["qkv", "proj", "fc1", "fc2", "conv_after_body", "conv_last"],
        )
        config = LoraConfig(
            r=int(self.lora_cfg.get("r", 16)),
            lora_alpha=int(self.lora_cfg.get("lora_alpha", 16)),
            target_modules=list(target_modules),
            bias=str(self.lora_cfg.get("bias", "none")),
        )
        net = get_peft_model(net, config)
        trainable, total = net.get_nb_trainable_parameters()
        logger.info(f"LoRA enabled: trainable={trainable:,} / total={total:,} ({100 * trainable / max(total, 1):.2f}%)")
        return net

    def init_training_settings(self) -> None:
        """Same as SRModel but never builds EMA when LoRA is on (peft-incompatible)."""
        if self.lora_enabled and float(self.opt["train"].get("ema_decay", 0) or 0) > 0:
            logger = get_root_logger()
            logger.warning("LoRA is enabled: forcing ema_decay=0 (EMA + peft not supported).")
            self.opt["train"]["ema_decay"] = 0
        super().init_training_settings()

    def setup_optimizers(self) -> None:
        train_opt = self.opt["train"]
        optim_params = []
        frozen = 0
        for _k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                frozen += 1
        if frozen and self.lora_enabled:
            logger = get_root_logger()
            logger.info(f"LoRA: skipping {frozen} frozen base parameters in optimizer.")
        elif frozen:
            logger = get_root_logger()
            for k, v in self.net_g.named_parameters():
                if not v.requires_grad:
                    logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt["optim_g"])
        self.optimizers.append(self.optimizer_g)

    def _merged_base_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return a CPU state_dict matching plain TSFDANSwinIR keys (LoRA merged)."""
        bare = self.get_bare_model(self.net_g)
        if not _is_peft_model(bare):
            return OrderedDict((k, v.detach().cpu().clone()) for k, v in bare.state_dict().items())

        # merge_and_unload is destructive — deepcopy so training adapters stay intact.
        merged = deepcopy(bare).merge_and_unload()
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in merged.state_dict().items())

    def save_network(
        self,
        net: nn.Module | list[nn.Module],
        net_label: str,
        current_iter: int,
        param_key: str | list[str] = "params",
    ) -> None:
        """Save merged base weights under ``params`` for TSFDAN infer scripts."""
        if not self.lora_enabled:
            super().save_network(net, net_label, current_iter, param_key)
            return

        import os
        import time

        if current_iter == -1:
            current_iter_name: int | str = "latest"
        else:
            current_iter_name = current_iter
        save_filename = f"{net_label}_{current_iter_name}.pth"
        save_path = os.path.join(self.opt["path"]["models"], save_filename)

        # Only export the generator as merged base weights (single param_key).
        save_dict = {"params": self._merged_base_state_dict()}
        logger = get_root_logger()
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as exc:  # noqa: BLE001 - match BaseModel retry style
                logger.warning(f"Save model error: {exc}, remaining retry times: {retry - 1}")
                time.sleep(1)
            else:
                logger.info(f"Saved merged LoRA weights to {save_path}")
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f"Still cannot save {save_path}. Just ignore it.")

    def save(self, epoch: int, current_iter: int) -> None:
        self.save_network(self.net_g, "net_g", current_iter, param_key="params")
        self.save_training_state(epoch, current_iter)
