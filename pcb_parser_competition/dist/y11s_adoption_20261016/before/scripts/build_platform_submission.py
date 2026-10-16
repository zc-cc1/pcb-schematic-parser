"""Build a self-contained source ZIP without training data or local caches."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT/'submission/platform_adapter'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script_closure():
    needed, pending = set(), ['run_observed_net_fields.py']
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        path = ROOT/'scripts'/name
        source = path.read_text(encoding='utf-8-sig')
        needed.add(name)
        modules = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module.split('.')[0])
            elif isinstance(node, ast.Import):
                modules.extend(item.name.split('.')[0] for item in node.names)
        pending.extend(module+'.py' for module in modules if (ROOT/'scripts'/(module+'.py')).is_file())
        pending.extend(re.findall(r'scripts/([a-zA-Z0-9_]+\.py)', source))
    return sorted(needed)


def copy(relative, target):
    source, destination = ROOT/relative, target/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='20261015')
    parser.add_argument('--archive-name', default='submission.zip')
    parser.add_argument('--runtime-profile', choices=['current-cu128', 'official-cu121'], default='current-cu128')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Tag must contain letters, numbers, hyphens or underscores')
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.zip', args.archive_name):
        parser.error('Archive name must be a simple .zip filename')
    # New build directories retain earlier reviewable packages; no source deletion.
    sequence = 1
    while (ROOT/f'submission/platform_test_{args.tag}_{sequence}').exists():
        sequence += 1
    target = ROOT/f'submission/platform_test_{args.tag}_{sequence}'
    target.mkdir(parents=True)
    scripts = script_closure()
    runtime_sources = []
    for name in scripts:
        relative = Path('scripts')/name
        copy(relative, target); runtime_sources.append(relative)
    for path in sorted((ROOT/'src').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
            relative = path.relative_to(ROOT)
            copy(relative, target)
            if path.suffix == '.py':
                runtime_sources.append(relative)
    config_names = set()
    for relative in runtime_sources:
        source = (ROOT/relative).read_text(encoding='utf-8-sig')
        config_names.update(re.findall(r'configs/([a-zA-Z0-9_.-]+\.json)', source))
        # Also cover ROOT/'configs'/name with a list of bare filenames. The
        # output guard loads both v1 and v2 to verify prediction provenance.
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.endswith('.json') and '/' not in node.value and '\\' not in node.value
                    and (ROOT/'configs'/node.value).is_file()):
                config_names.add(node.value)
    config_names.add('baseline_v3_adopted.json')
    for name in sorted(config_names):
        copy(Path('configs')/name, target)
    for name in ['detector_dataset300_v3_nano.pt', 'unet_wire_pin_v1.pt', 'junction_classifier_v1.pt',
                 'HR-DONK_LICENSE.txt', 'baseline_provenance.json']:
        copy(Path('weights')/name, target)
    library = json.loads((ROOT/'configs/component_pin_library_v1.json').read_text(encoding='utf-8'))
    copy(Path(library['library_path']), target)
    for part in json.loads((ROOT/library['library_path']).read_text(encoding='utf-8'))['parts']:
        copy(Path(part['source']['local_path']), target)
    for name in ['agent.py', 'adapter.py', 'component_export.py', 'Dockerfile', 'run.sh', 'requirements.txt']:
        shutil.copy2(TEMPLATES/name, target/name)
    if args.runtime_profile == 'official-cu121':
        shutil.copy2(TEMPLATES/'Dockerfile.official_slim', target/'Dockerfile')
    ocr_package = Path(importlib.util.find_spec('rapidocr').origin).parent
    ocr_target = target/'assets/ocr'; ocr_target.mkdir(parents=True)
    for name in ['PP-OCRv6_det_small.onnx', 'PP-OCRv6_rec_small.onnx', 'ch_ppocr_mobile_v2.0_cls_mobile.onnx']:
        shutil.copy2(ocr_package/'models'/name, ocr_target/name)
    matplotlib = Path(importlib.util.find_spec('matplotlib').origin).parent
    fonts = target/'assets/fonts'; fonts.mkdir(parents=True)
    for name in ['DejaVuSansMono.ttf', 'LICENSE_DEJAVU']:
        shutil.copy2(matplotlib/'mpl-data/fonts/ttf'/name, fonts/name)
    patches = []
    for relative in runtime_sources:
        path = target/relative
        source = path.read_text(encoding='utf-8-sig')
        # scripts/foo.py -> parents[1], src/pcb_parser/foo.py -> parents[2]
        expression = f"str(Path(__file__).resolve().parents[{len(relative.parts)-1}] / 'assets/fonts/DejaVuSansMono.ttf')"
        source, count = re.subn(r'''(['"])C:/Windows/Fonts/[^'"]+\.(?:ttf|ttc)\1''', lambda _: expression, source)
        if count:
            patches.append({'file': relative.as_posix(), 'change': 'bundled_portable_font', 'replacements': count})
        if relative.as_posix() == 'src/pcb_parser/pin_text_ocr.py':
            old = "paths={role:package/'models'/name for role,name in names.items()}"
            new = "paths={role:Path(__file__).resolve().parents[2]/'assets'/'ocr'/name for role,name in names.items()}"
            if old not in source:
                raise ValueError('Expected OCR asset locator changed; review packaging patch')
            source = source.replace(old, new)
            patches.append({'file': relative.as_posix(), 'change': 'bundled_offline_ocr_models'})
        path.write_text(source, encoding='utf-8', newline='\n')
        compile(source, str(path), 'exec')
    for name in ['agent.py', 'adapter.py', 'component_export.py', 'Dockerfile', 'run.sh', 'requirements.txt']:
        path = target/name
        path.write_text(path.read_text(encoding='utf-8-sig'), encoding='utf-8', newline='\n')
        if path.suffix == '.py':
            compile(path.read_text(encoding='utf-8'), str(path), 'exec')
    readme = '''# 平台测试提交包

这是现有完整 S24 流程的源码包：YOLO v3 检测 → RapidOCR → 冻结 U-Net v1 → 引脚归属、公共引脚库与字段补全 → 工程整网出口。
保留项目默认 detector_dataset300_v3_nano.pt 和 unet_wire_pin_v1.pt；10月15日的 YOLO11、U-Net v7 与独立线段解码实验尚未接入该完整流程，本包没有擅自替换默认权重。

## 上传

优先在本机构建、测试并导出 submission_target.zip，再上传该预构建镜像ZIP；此源码ZIP用于本机构建。
选择文件后平台可能立即开始提交。本包不使用大模型 API，无需密钥，Token 全部为 0。
GPU 默认自动选择：先执行真实 CUDA 矩阵运算，成功后选择 cuda:0；未开放GPU、驱动或架构不兼容时记录具体错误并回退 CPU。不更改 CUDA_VISIBLE_DEVICES，也不强制要求 GPU 以免平台缺设备时整批失败。

## 输出

- /output/final_results/<case_id>_Pred.json：默认使用实际 2026-10-08 评分协议的嵌套 pins、hyperGraph、edges 布局。
- /output/project_results/<case_id>_Pred.json：保留项目原始嵌套 pins、hyperGraph、edges 布局，便于排查。
- /output/final_results/decision_log.json 与 /output/decision_log.json：同时写两处；官方 README 和平台网页对该位置存在不一致。
- /output/logs/：每张图的运行日志、覆盖统计和字段来源。

value、H-y 坐标系、引脚坐标和网络成员保持。project_results 保留全部原始模型/OCR结果。
依据2026-10-16官方公开标注，r/c/l的Name使用已识别位号；已确认的重复位号实例保留原始基础名称。
r/c/l/d/led的评分框采用公开训练集一致使用的尺寸模板，保留模型预测中心与长轴方向。
模板短边/长边分别为r:8/20、c:16/20、l:6/34、d和led:14/20像素；不是按测试图片查表回填答案。
IC、二极管、LED的已有Name保持，200训练标注与10个Golden Case在IC Name等字段上存在版本差异。
赛题PDF第13、16、18、19页的JSON截图确认了components、嵌套pins和hyperGraph/edges结构，以及Name字段的大写N。
PDF示例使用ic/resistor等长类别名，与本次实际评分报告及平台页面的r编码存在冲突；不能把示例当作完整的当前类别表。
resistor/capacitor/inductor/diode/ic/led 对应 r/c/l/d/box/led；提交8180的145个匹配元件和公开标注已验证这六类，不代表覆盖全部类别。
sample/project 仅作显式调试选项；平台部署默认 target。显存不少于24 GiB时自动最多3例并行，小显存GPU和CPU默认串行；可用 --workers 1/2/3 覆盖。每例运行目录与配置缓存隔离。

## 本地运行

Docker（在解压目录执行；输入/输出路径使用绝对路径）：

    docker build -t pcb-parser:test .
    docker run --rm --gpus all -v <输入绝对路径>:/input:ro -v <输出绝对路径>:/output:rw pcb-parser:test -input /input -output /output

CPU 测试可移除 --gpus all，并添加 -e PCB_DEVICE=cpu（放在镜像名称之前）。
基础镜像 edarace-agent-base:ubuntu24 来自官方示例；本机需能获取该镜像，平台是否预置须看构建日志。
构建时从 PyTorch 官方索引安装 CUDA 12.8 的 torch 2.10.0 / torchvision 0.25.0，推理所需 YOLO/U-Net/OCR 权重均已随包带入。

已安装同版本依赖的 Python 也能运行：

    python agent.py -input <输入目录> -output <输出目录> --device cpu

PNG 可位于子目录，保留文件名的前导零。不同目录若出现同名 PNG 会报错，避免覆盖结果。
每例生成独立日志；部分图片失败时输出空 JSON 并继续，其失败原因计入 decision_log。全部失败或预检失败返回非零退出码。
默认清理临时大概率图；需要保留时添加 --keep-work。平台可能限制中间文件下载，日志也会在失败时打印关键末尾信息。

## 验证边界

上一版镜像8422已获28.92223分。本次包含无位号电源/地出口、LED文字与框、IC Name优先级和IC实框定位修复，具体构建/测试记录随新归档保存，尚需平台复测；不能沿用上一版分数或把离线诊断当成官方成绩。
原有完整网覆盖仍有限，工程编号规则尚未由官方 Golden 确认。打包通过不代表取得官方识别成绩。
不含训练图片、参考 JSON、预测缓存、训练过程、Windows 虚拟环境或 API 密钥。
package_manifest.json 列出逐文件哈希、模型、配置与仅提交副本中的可移植性修改。
'''
    readme += '\n## 电源元件出口修复\n\n当前图像已检测的地/电源不再因缺位号而全部丢弃，供电类别转换为官方v。技术检测ID仅用于JSON唯一标识，评分忽略电源key/Name。value只保留已识别文字，不能确认时保持null；不根据符号类别统一猜GND。原有元件、引脚和网络保持。project_results保留原流程输出，component_results保留补齐电源后的转换输入。logs包含电源导出证据与缺失字段附近OCR，runtime_environment.json记录实际设备选择。\n'
    readme += '\n## LED 文字与框修复\n\n同一字形的高置信度完整LEDn与截断EDn读法冲突时，不再将截断副本当作型号。LED评分框方向由当前图像两侧引线确定；中心仅在两个局部YOLO视野定位一致、不跨邻近LED归属、且偏移超过2.5px时修正，证据不足保留原坐标。固定公开尺寸仍为14×20或20×14。其余元件、文字、引脚、网络不受框修复影响。logs/<case>_led_geometry.json记录前后框、局部检测候选、方向证据与耗时；project_results与component_results保留几何修复前结果。公开数据诊断和受控偏移测试不代表线上失败框已恢复。\n'
    readme += '\n## IC Name 候选优先级\n\n保留全部已有OCR行及观察读法到logs/<case>_ocr_texts.json；按可信型号、实际观察到的自身位号、其他器件说明选择IC Name，来源、附近候选和拒绝原因见logs/<case>_ic_name_selection.json。已绑定型号保留；不把引脚/电源标签、其他器件位号或技术检测ID用作名称，不从字典补造字符。该步骤仅作用于最终Target的IC Name，project_results、component_results及原有引脚/网络保持。型号缺失时U1会优先于2.4GHz、T_V等说明，这是用户选择的排序，尚不保证适合线上不同标注版本。\n'
    readme += '\n## IC 实框定位\n\n最终Target的IC bbox从当前PNG中的四条直边和四角证据定位，替代检测框的普遍外扩；不按型号查固定尺寸，也不按线上误差猜平移。小于40px短边、缺边、多框竞争或不连续边界保留原框。logs/<case>_ic_geometry.json记录前后框、四边支持、候选和拒绝原因及耗时。原始project_results、component_results及Name、引脚、网络均不被几何步骤修改。公开配对诊断不是新官方分数，线上五个失败实例尚需新版复测。\n'
    if args.runtime_profile == 'official-cu121':
        readme = readme.replace('构建时从 PyTorch 官方索引安装 CUDA 12.8 的 torch 2.10.0 / torchvision 0.25.0',
            '复用用户提供的官方 edarace-agent-base:ubuntu24 中 CUDA 12.1 的 torch 2.5.1 / torchvision 0.20.1，不叠加安装第二套CUDA')
        readme += '\n## 官方运行时小包\n\n本运行时面向比赛A800（sm_80），不支持本机RTX5060（sm_120）的GPU算子；本机需使用 --device cpu 验证。原cu128 Docker镜像保留用于RTX5060本地GPU测试。模型权重相同，结果仍须重新验证。\n'
    (target/'README.md').write_text(readme, encoding='utf-8', newline='\n')
    adapter_tree = ast.parse((target/'adapter.py').read_text(encoding='utf-8'))
    component_policy = next(node.value.value for node in adapter_tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and any(isinstance(item, ast.Name) and item.id == 'COMPONENT_OUTPUT_POLICY' for item in node.targets))
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'pipeline': 'S24_existing_complete_image_pipeline_engineering_export',
                'output_format': 'target', 'evaluation_protocol': '2026-10-08-global-power-rails',
                'type_mapping_status': 'six aliases verified in submission 8180 and public labels',
                'component_output_policy': component_policy,
                'runtime_profile': args.runtime_profile,
                'original_output_format_preserved_under': '/output/project_results',
                'coordinate_origin': 'bottom_left_H_minus_y', 'scripts': scripts,
                'configs': sorted(config_names), 'portability_patches': patches,
                'training_data_included': False, 'api_keys_included': False,
                'docker_build_verified': False, 'files': {}}
    for path in sorted(target.rglob('*')):
        if path.is_file():
            relative = path.relative_to(target).as_posix()
            manifest['files'][relative] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    manifest['runtime_assets'] = [name for name in manifest['files'] if name.startswith(('weights/', 'assets/', 'data/'))]
    (target/'package_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    dist = ROOT/'dist'; dist.mkdir(exist_ok=True)
    archive = dist/args.archive_name
    temporary = archive.with_suffix('.zip.tmp')
    # A fixed 1980 timestamp can make BuildKit reuse an older same-size file
    # when transferring successive contexts. Give each source snapshot its
    # actual build time, and verify the manifest inside the built image too.
    zip_timestamp = datetime.now().timetuple()[:6]
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(target.rglob('*')):
            if not path.is_file():
                continue
            name = path.relative_to(target).as_posix()
            entry = zipfile.ZipInfo(name, date_time=zip_timestamp)
            entry.create_system = 3
            mode = 0o755 if name == 'run.sh' else 0o644
            entry.external_attr = (stat.S_IFREG | mode) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(entry, path.read_bytes(), compresslevel=6)
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
        names = bundle.namelist()
        assert {'Dockerfile', 'agent.py', 'run.sh', 'package_manifest.json'} <= set(names)
        assert (bundle.getinfo('run.sh').external_attr >> 16) & stat.S_IXUSR
        assert b'\r' not in bundle.read('run.sh')
        assert not any(name.startswith(('.venv', 'runs/', 'data/standardized', 'data/detection', 'data/splits')) for name in names)
    result = {'source_directory': str(target), 'archive': str(archive), 'bytes': archive.stat().st_size,
              'sha256': sha(archive), 'file_count': len(names), 'scripts': len(scripts),
              'crc_verified': True, 'run_sh_lf_and_executable': True, 'docker_build_verified': False}
    report_name = 'package_report.json' if args.archive_name == 'submission.zip' else archive.stem+'_report.json'
    (dist/report_name).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
