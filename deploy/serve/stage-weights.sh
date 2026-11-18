#!/usr/bin/env bash
# Copy the checkpoints the customer image bakes in. Official SwinIR / TSFDAN
# files already live under experiments/pretrained_models and are not copied here.
set -euo pipefail

dest="$(cd "$(dirname "$0")" && pwd)/weights"
data="/mnt/home/huang/cst_ai"
mkdir -p "$dest"

copy_file() {
    local src="$1"
    local name="$2"
    if [[ ! -f "$src" ]]; then
        printf 'missing weight: %s\n' "$src" >&2
        exit 1
    fi
    cp -f "$src" "${dest}/${name}"
}

copy_file \
    "${data}/experiments/aligned_drive/train_SwinIR_grayDN_paired_tiff_noise25_P128W8/models/net_g_latest.pth" \
    "aligned_drive_net_g_latest.pth"
copy_file \
    "${data}/experiments/cylinder_260520_gt90ua/train_SwinIR_grayDN_cylinder_gt90ua_noise25_P128W8_bs4_i30k/models/net_g_latest.pth" \
    "cylinder90_net_g_latest.pth"
copy_file \
    "${data}/experiments/cylinder_phantom_mix/train_SwinIR_grayDN_cylinder_phantom_mix_noise25_P128W8_bs4_i30k/models/net_g_latest.pth" \
    "mix_net_g_latest.pth"
copy_file \
    "${data}/experiments/alcast_150kv_gt66ma/train_SwinIR_grayDN_alcast_gt66ma_noise25_P128W8_bs4_i30k/models/net_g_latest.pth" \
    "alcast_net_g_latest.pth"
copy_file \
    "${data}/archive/cylinderblock_2673001900/net_g_latest.pth" \
    "archive_cylinderblock_net_g_latest.pth"
copy_file \
    "${data}/experiments/cylinder_phantom_mix_tsfdan_lora/train_TSFDAN_lora_cylinder_phantom_mix_P128_bs4_i10k/models/net_g_latest.pth" \
    "tsfdan_lora_mix_net_g_latest.pth"
copy_file \
    "${data}/demo_models/SwinIR_Cylinderhead_mean3_20261105_054425_JST_final_i30000.pth" \
    "cylinderhead_final_i30000.pth"
copy_file \
    "${data}/experiments/cylinderhead_300kv_mean3_20261104_173035_JST/control/train.yml" \
    "cylinderhead_train.yml"

find "$dest" -type f -printf '%s %f\n' | sort -n
