"""运行第2～16周的30个Python无硬件示例，并检查主要教学结果。会更新各周outputs。"""
from pathlib import Path
import subprocess
import sys
import json
import platform
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    rows=[]
    for folder in sorted(ROOT.iterdir()):
        if not folder.is_dir() or not folder.name[:2].isdigit() or folder.name.startswith('01_'):
            continue
        for name in ['demo.py','exercise_solution.py']:
            script=folder/name
            try:
                done=subprocess.run([sys.executable,str(script)],cwd=ROOT,capture_output=True,
                                    text=True,encoding='utf-8',errors='replace',timeout=120,
                                    env=dict(__import__('os').environ,PYTHONIOENCODING='utf-8'))
                passed=done.returncode==0
                row={'script':str(script.relative_to(ROOT)),'passed':passed}
                if not passed:
                    row['error']=(done.stdout+chr(10)+done.stderr)[-6000:]
            except subprocess.TimeoutExpired:
                row={'script':str(script.relative_to(ROOT)),'passed':False,'error':'运行超过120秒'}
            rows.append(row)
            print(('PASS ' if row['passed'] else 'FAIL ')+str(script.relative_to(ROOT)),flush=True)
    checks=[]
    def metric(folder):
        return json.loads((ROOT/folder/'outputs/metrics.json').read_text(encoding='utf-8'))
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
    try:
        check('三种独立形状',metric('06_edges_contours')['count']==3)
        check('红色目标计数',metric('07_color_detection')['red_count']==3)
        check('视频完整读取80帧',metric('08_video')['processed_frames']==80)
        check('模板定位',metric('09_template')['best_top_left']==[170,90])
        check('单应矩阵角点误差低于5像素',metric('10_features')['mean_corner_error_pixels']<5)
        dx,dy=metric('11_motion')['median_displacement']
        check('光流位移接近12像素',abs(dx-12)<1 and abs(dy)<1)
        check('理想标定重投影误差低于0.01像素',metric('12_calibration')['rms_pixels']<.01)
        check('粘连目标分成两个区域',metric('13_segmentation')['regions']==2)
        check('合成形状分类准确率至少90%',metric('14_ml')['accuracy']>=.9)
        m=metric('15_inspection_project')
        check('六个零件一个缺陷',m['count']==6 and m['defects']==1)
        check('合成文档扫描成功',metric('16_scanner_project')['success'])
    except (OSError,KeyError,ValueError) as error:
        check('所有指标文件可读取: '+str(error),False)
    for row in checks:
        print(('PASS ' if row['passed'] else 'FAIL ')+row['check'],flush=True)
    passed=all(r['passed'] for r in rows+checks) and len(rows)==30
    result={'all_passed':passed,'python':platform.python_version(),'opencv':cv2.__version__,
            'numpy':np.__version__,'scripts':rows,'behavior_checks':checks,
            'scope':'第2～16周Python，不包括第1周C++',
            'not_tested':['第1周C++，请单独运行CTest','真实摄像头','自己的棋盘照片标定','真实工业图片泛化能力']}
    (ROOT/'验证结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if not passed:
        raise SystemExit(1)

if __name__=='__main__':
    main()
