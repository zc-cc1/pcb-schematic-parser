"""Read-only visualization of raw U-Net channels, before downstream processing."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--probabilities', type=Path, required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if any(path.resolve().is_relative_to(a.output.resolve()) for path in [a.image, a.probabilities, a.metadata]):
        raise ValueError('Review output overlaps source inputs')
    with Image.open(a.image) as im:rgb = np.asarray(im.convert('RGB'))
    meta = json.loads(a.metadata.read_text(encoding='utf-8'))
    cfg = json.loads((ROOT/'configs/unet_inference_v2.json').read_text(encoding='utf-8'))
    if (meta['image_sha256'] != sha(a.image) or meta['size'] != [rgb.shape[1], rgb.shape[0]]
        or meta['coordinate_origin'] != 'top_left' or meta['weights_sha256'] != cfg['checkpoint_sha256']):
        raise ValueError('U-Net prediction provenance differs from PNG/config')
    source_hashes = {str(path.resolve()): sha(path) for path in [a.image, a.probabilities, a.metadata]}
    with np.load(a.probabilities, allow_pickle=False) as data:
        wire, tip = data['wire'].copy(), data['pin_tip'].copy()
    for channel in [wire, tip]:
        if (channel.shape != rgb.shape[:2] or not np.isfinite(channel).all()
            or channel.min() < 0 or channel.max() > 1):raise ValueError('Invalid raw probability channel')
    a.output.mkdir(parents=True, exist_ok=True)
    for name, channel, threshold in [('wire', wire, cfg['wire_threshold']), ('pin_tip', tip, cfg['pin_threshold'])]:
        Image.fromarray(np.rint(channel*255).astype(np.uint8)).save(a.output/f'{name}_probability_gray.png')
        Image.fromarray((channel >= threshold).astype(np.uint8)*255).save(a.output/f'{name}_mask_t{threshold:.2f}.png')
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    case = a.image.stem
    overlay = np.zeros((*wire.shape, 4), dtype=np.float32)
    overlay[..., :3] = [0.05, .43, .95]
    overlay[..., 3] = (wire >= cfg['wire_threshold'])*.90
    fig, axes = plt.subplots(2, 1, figsize=(12, 11.2), dpi=160)
    axes[0].imshow(rgb, interpolation='nearest');axes[0].set_title(f'{case} · 原始 PNG', fontsize=16, pad=12)
    axes[1].imshow(rgb, interpolation='nearest');axes[1].imshow(overlay, interpolation='nearest')
    axes[1].set_title(f"原图 + U-Net 导线分割（蓝色：模型分数 ≥ {cfg['wire_threshold']:.2f}）", fontsize=16, pad=12)
    for ax in axes:ax.axis('off')
    fig.subplots_adjust(left=.015, right=.985, top=.965, bottom=.035, hspace=.13)
    fig.savefig(a.output/'wire_compare.png', facecolor='white');plt.close(fig)
    for name, channel, title in [('wire', wire, '导线通道'), ('pin_tip', tip, '引脚 tip 通道')]:
        fig, ax = plt.subplots(figsize=(12, 5.9), dpi=160)
        image = ax.imshow(channel, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{case} · U-Net 原始{name}热图（{title}）', fontsize=16, pad=12)
        ax.axis('off');fig.colorbar(image, ax=ax, fraction=.025, pad=.015, label='模型输出分数（0～1）')
        fig.tight_layout();fig.savefig(a.output/f'{name}_heatmap.png', facecolor='white');plt.close(fig)
    h, w = rgb.shape[:2]
    summary = {'case_id': case, 'size': [w, h], 'image_sha256': meta['image_sha256'],
        'weights_sha256': meta['weights_sha256'], 'source_input_sha256': source_hashes,
        'scale': 1.0, 'channels': ['wire', 'pin_tip'], 'wire_threshold': cfg['wire_threshold'],
        'pin_candidate_threshold': cfg['pin_threshold'], 'raw_tip_candidates_in_source': len(meta['pin_candidates']),
        'raw_probability_arrays_modified': False, 'model_or_inference_config_modified': False,
        'component_body_mask_applied': False, 'skeletonization_applied': False,
        'component_ownership_OR_OCR_OR_network_export_applied': False,
        'display_gray_probability_quantization': 'round(model_score*255), display only; raw NPZ unchanged',
        'artifacts': ['wire_compare.png', 'wire_heatmap.png', 'pin_tip_heatmap.png',
                      'wire_probability_gray.png', 'pin_tip_probability_gray.png',
                      f"wire_mask_t{cfg['wire_threshold']:.2f}.png", f"pin_tip_mask_t{cfg['pin_threshold']:.2f}.png"]}
    if any(sha(Path(path)) != digest for path, digest in source_hashes.items()):raise ValueError('Source changed during rendering')
    (a.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print({k: summary[k] for k in ['case_id', 'size', 'scale', 'wire_threshold', 'pin_candidate_threshold', 'raw_tip_candidates_in_source']})


if __name__ == '__main__':main()
