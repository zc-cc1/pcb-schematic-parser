"""Read-only 0036 wire-channel crops and score profiles for user inspection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from render_unet_review import ROOT, sha


def main():
    p = argparse.ArgumentParser();p.add_argument('--review', type=Path, default=ROOT/'runs/unet_review_0036');a = p.parse_args()
    review = json.loads((a.review/'summary.json').read_text(encoding='utf-8'))
    if review['case_id'] != '0036' or review['scale'] != 1.0:raise ValueError('Region coordinates are for original 0036 PNG')
    sources = review['source_input_sha256']
    if any(sha(Path(path)) != h for path, h in sources.items()):raise ValueError('Frozen probability/image inputs changed')
    image_path = next(Path(path) for path in sources if Path(path).suffix == '.png')
    probability_path = next(Path(path) for path in sources if Path(path).suffix == '.npz')
    with Image.open(image_path) as source:rgb = np.asarray(source.convert('RGB'))
    with np.load(probability_path, allow_pickle=False) as source:wire = source['wire'].copy()
    threshold = review['wire_threshold']
    # Coordinates are visually selected on the original PNG for explanation.
    # They never enter model inference, training or a topology repair.
    regions = [
        {'name': 'X1 左侧引线', 'box': [536, 27, 795, 119], 'y': 80, 'x_start': 615, 'x_end': 692},
        {'name': 'U1 第 5 脚 DOUT 引线', 'box': [535, 136, 781, 237], 'y': 181, 'x_start': 604, 'x_end': 632},
    ]
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    fig, axes = plt.subplots(4, 2, figsize=(12.6, 11), dpi=160,
                             gridspec_kw={'height_ratios': [1, 1, 1, .88]}, layout='constrained')
    stats = [];heat = None
    for col, region in enumerate(regions):
        x1, y1, x2, y2 = region['box'];y = region['y'];left, right = region['x_start'], region['x_end']
        crop = wire[y1:y2, x1:x2]
        profile = wire[y-1:y+2, left:right+1].max(axis=0)
        xs = np.arange(left, right+1)
        row = {**region, 'sampling': 'per-column maximum across original y-1,y,y+1',
            'minimum': float(profile.min()), 'median': float(np.median(profile)), 'maximum': float(profile.max()),
            'column_count': len(profile), 'columns_at_or_above_050': int((profile >= .5).sum()),
            'columns_at_or_above_030': int((profile >= .3).sum()),
            'profile': [{'x': int(x), 'wire_score': float(v)} for x, v in zip(xs, profile)]}
        stats.append(row)
        axes[0, col].imshow(rgb[y1:y2, x1:x2], interpolation='nearest')
        axes[0, col].add_patch(Rectangle((left-x1-.5, y-y1-1.5), right-left+1, 3,
            facecolor='none', edgecolor='#d87500', linewidth=1.1))
        axes[0, col].set_title(region['name']+'\n① 原图（橙框：取样区域）', fontsize=15, pad=10)
        heat = axes[1, col].imshow(crop, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        axes[1, col].set_title('② 原始导线分数：亮色高，暗色低', fontsize=14, pad=8)
        axes[2, col].imshow(crop >= threshold, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[2, col].set_title(f'③ 当前分割掩膜（分数 ≥ {threshold:.2f}）', fontsize=14, pad=8)
        for ax in axes[:3, col]:ax.axis('off')
        ax = axes[3, col]
        ax.plot(xs, profile, color='#2463c5', linewidth=1.8, label='取样区域的模型分数')
        ax.axhline(threshold, color='#c54040', linewidth=1.5, linestyle='--', label=f'当前阈值 {threshold:.2f}')
        ax.set_xlim(left, right);ax.set_ylim(0, 1);ax.grid(axis='y', color='#dddddd', linewidth=.6)
        ax.set_xlabel('原图 x 坐标（px）', fontsize=12);ax.set_ylabel('导线分数', fontsize=12)
        ax.set_title(f"④ 分数曲线：中位数 {row['median']:.3f}", fontsize=14, pad=8)
        ax.tick_params(labelsize=11);ax.legend(loc='upper right', fontsize=10, frameon=False)
    fig.colorbar(heat, ax=list(axes[1]), shrink=.85, pad=.01, fraction=.015, label='模型分数 0～1')
    fig.suptitle('0036 · U-Net 原尺寸导线输出：两处漏分割位置', fontsize=19)
    fig.savefig(a.review/'wire_missing_regions.png', facecolor='white');plt.close(fig)
    data = {'case_id': '0036', 'image_sha256': review['image_sha256'], 'weights_sha256': review['weights_sha256'],
        'scale': 1.0, 'wire_threshold': threshold, 'source_input_sha256': sources,
        'model_or_threshold_changed': False, 'used_in_inference': False, 'used_in_training': False,
        'scores_are_not_accuracy': True, 'regions': stats}
    if any(sha(Path(path)) != h for path, h in sources.items()):raise ValueError('Source changed during render')
    (a.review/'wire_missing_regions.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print([{k: r[k] for k in ['name', 'median', 'maximum', 'column_count', 'columns_at_or_above_050', 'columns_at_or_above_030']} for r in stats])


if __name__ == '__main__':main()
