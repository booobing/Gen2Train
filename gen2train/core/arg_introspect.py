"""sd-scripts의 argparse 정의를 그대로 읽어(introspection) 고급 파라미터 탭을 자동 생성하기 위한 메타데이터를 뽑아낸다.

새 kohya_ss 버전으로 sd-scripts를 갱신해도, 이 파일은 setup_parser()가 반환하는
argparse.ArgumentParser 객체만 들여다보므로 별도 유지보수 없이 새 옵션이 자동으로 잡힌다.
"""
import argparse
import importlib
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from .. import settings

# 모델 타입 -> sd-scripts 진입 스크립트(모듈명, argparse 소스)
SCRIPT_MODULES = {
    "sd": "train_network",
    "sdxl": "sdxl_train_network",
}

# 데이터셋/모델 준비 단계에서 Gen2Train이 자체적으로 값을 채우므로 고급 탭에 노출할 필요가 없는 인자.
HIDDEN_DESTS = {
    "help",
    "pretrained_model_name_or_path",
    "train_data_dir",
    "reg_data_dir",
    "output_dir",
    "output_name",
    "dataset_config",
    "config_file",
    "console_log_file",
    "network_module",  # Gen2Train은 표준 LoRA(networks.lora)만 지원, 항상 고정값으로 채움
    "network_args",  # LyCORIS 등 확장 네트워크 전용 옵션 - MVP 범위 밖
    "caption_extension",  # dataset_prep.py가 쓰는 캡션 확장자(.txt)와 항상 일치해야 하므로 trainer가 고정 전달
}

# dest 접두어 -> UI 그룹 라벨. 먼저 매치되는 규칙이 우선한다.
GROUP_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("모델", ("v2", "v_parameterization", "sdxl", "tokenizer_cache_dir", "disable_mmap")),
    (
        "LoRA 네트워크",
        (
            "network_",
            "unet_lr",
            "text_encoder_lr",
            "dim_from_weights",
            "training_comment",
            "scale_weight_norms",
            "base_weights",
            "no_metadata",
            "save_model_as",
            "cpu_offload_checkpointing",
            "fp8_base",
        ),
    ),
    (
        "데이터셋 / 캡션",
        (
            "resolution",
            "enable_bucket",
            "bucket_",
            "min_bucket_reso",
            "max_bucket_reso",
            "random_crop",
            "color_aug",
            "flip_aug",
            "face_crop_aug_range",
            "caption_",
            "shuffle_caption",
            "keep_tokens",
            "token_warmup",
            "dataset_repeats",
            "in_json",
            "max_token_length",
            "cache_latents",
            "cache_info",
            "alpha_mask",
            "secondary_separator",
            "weighted_captions",
        ),
    ),
    ("저장", ("save_", "resume", "huggingface_")),
    ("샘플 이미지", ("sample_",)),
    (
        "노이즈 / 손실 함수",
        (
            "noise_offset",
            "multires_noise",
            "adaptive_noise_scale",
            "ip_noise_gamma",
            "zero_terminal_snr",
            "min_snr_gamma",
            "debiased_estimation_loss",
            "v_pred_like_loss",
            "loss_type",
            "huber_",
            "masked_loss",
            "prior_loss_weight",
        ),
    ),
    (
        "옵티마이저 / LR 스케줄러",
        ("optimizer", "lr_", "max_grad_norm"),
    ),
    (
        "학습 / 성능",
        (
            "xformers",
            "sdpa",
            "mem_eff_attn",
            "gradient_",
            "mixed_precision",
            "full_fp16",
            "full_bf16",
            "fp8_",
            "highvram",
            "lowvram",
            "persistent_data_loader_workers",
            "max_data_loader_n_workers",
            "seed",
            "clip_skip",
            "vae",
            "max_train_",
            "train_batch_size",
            "logging_dir",
            "log_",
            "wandb",
            "deepspeed",
            "ddp_",
            "dynamo_",
            "lowram",
            "num_cpu_threads_per_process",
        ),
    ),
]

DEFAULT_GROUP = "기타"


@dataclass
class ArgSpec:
    flag: str  # "--network_dim"
    dest: str  # "network_dim"
    kind: str  # "bool" | "int" | "float" | "choice" | "str"
    default: Any
    choices: Optional[list] = None
    help: str = ""
    group: str = DEFAULT_GROUP
    nargs: Optional[str] = None  # 예: "*" (multi-value)


def _infer_group(dest: str) -> str:
    for group, prefixes in GROUP_RULES:
        if any(dest == p or dest.startswith(p) for p in prefixes):
            return group
    return DEFAULT_GROUP


def _infer_kind(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    if action.choices:
        return "choice"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    return "str"


def _import_setup_parser(model_type: str):
    module_name = SCRIPT_MODULES[model_type]
    sd_scripts_dir = str(settings.SD_SCRIPTS_DIR)
    if sd_scripts_dir not in sys.path:
        sys.path.insert(0, sd_scripts_dir)
    module = importlib.import_module(module_name)
    return module.setup_parser


def get_arg_specs(model_type: str) -> list[ArgSpec]:
    """model_type: 'sd' 또는 'sdxl'. 해당 학습 스크립트가 지원하는 전체 CLI 인자를 반환한다."""
    setup_parser = _import_setup_parser(model_type)
    parser: argparse.ArgumentParser = setup_parser()

    specs: list[ArgSpec] = []
    seen_dests: set[str] = set()
    for action in parser._actions:
        dest = action.dest
        if dest in HIDDEN_DESTS or dest in seen_dests:
            continue
        if not action.option_strings:
            continue
        seen_dests.add(dest)
        flag = max(action.option_strings, key=len)
        specs.append(
            ArgSpec(
                flag=flag,
                dest=dest,
                kind=_infer_kind(action),
                default=action.default,
                choices=list(action.choices) if action.choices else None,
                help=action.help or "",
                group=_infer_group(dest),
                nargs=action.nargs if isinstance(action.nargs, str) else None,
            )
        )
    return specs


def group_order() -> list[str]:
    order = []
    for group, _ in GROUP_RULES:
        if group not in order:
            order.append(group)
    order.append(DEFAULT_GROUP)
    return order
