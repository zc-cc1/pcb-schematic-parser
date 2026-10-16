"""Finished BN and color diagnostics, with limits and visual evidence."""
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from unet_audit_common import ROOT,OUT,read,write,sha,assert_protected


def link(path,label=None):return f"[{label or path.name}]({str(path).replace(chr(92),'/')})"


def main():
    output=ROOT/'runs/unet_BN_order_diagnostic_v5';colorpath=ROOT/'runs/unet_color_response_v5'
    order=read(output/'evaluation.json');precise=read(output/'precise_evaluation.json');color=read(colorpath/'evaluation.json')
    old=read(ROOT/'runs/unet_negative_sampling_pilot_v4/evaluation.json')
    assert all(r['learned_weights_unchanged'] for r in order['estimation']+precise['estimation'])
    assert order['gradient_updates']==precise['gradient_updates']==color['gradient_updates']==0
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    names=[('exponential','指数平均'),('cumulative','FP32累计平均'),('precise_cumulative','FP64累计平均')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),dpi=160,layout='constrained')
    allpairs=order['pairs']+precise['pairs']
    for ax,arm in zip(axes,('control','negative_focus')):
        for dx,position,label in ((-.18,'forward','正序'),(.18,'reverse_batches','倒序')):
            values=[]
            for method,_ in names:
                source=precise if method=='precise_cumulative' else order
                values.append(source['summary'][f'{arm}__{method}__{position}']['wire']['dice']*100)
            bars=ax.bar(np.arange(3)+dx,values,width=.34,label=label)
            ax.bar_label(bars,fmt='%.4f',fontsize=8,padding=3)
        ax.set(xticks=np.arange(3),xticklabels=[s[1] for s in names],title=arm,ylabel='内部cal10全页Dice@0.5（%）',ylim=(76,80))
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('冻结可学习权重 · 只倒转同一批次的顺序 · 0次梯度更新')
    fig.savefig(output/'BN_order.png',facecolor='white');plt.close(fig)
    case='0093';box=(55,420,405,565);x0,y0,x1,y1=box
    rgb=np.asarray(Image.open(ROOT/f'data/standardized/{case}/{case}.png').convert('RGB'))
    gt=np.asarray(Image.open(ROOT/f'data/segmentation_unet_v1/{case}/wire_mask.png'))>0
    fig,axes=plt.subplots(2,3,figsize=(13,6.6),dpi=170,layout='constrained')
    axes[0,0].imshow(rgb[y0:y1,x0:x1]);axes[0,0].set_title('0093 原图：蓝色细线')
    over=rgb.copy();over[gt]=(.25*over[gt]+.75*np.array([20,180,95])).astype(np.uint8)
    axes[1,0].imshow(over[y0:y1,x0:x1]);axes[1,0].set_title('现有GT（整页未认证）')
    for ax,policy in zip((axes[0,1],axes[0,2],axes[1,1],axes[1,2]),('native','mean_gray','cycle_GBR','cycle_BRG')):
        mask=np.asarray(Image.open(colorpath/'evaluation'/case/policy/'wire_mask.png'))>0;over=rgb.copy()
        for region,tint in ((mask&gt,[20,180,95]),(mask&~gt,[25,140,235]),(~mask&gt,[230,55,60])):
            over[region]=(.25*over[region]+.75*np.array(tint)).astype(np.uint8)
        rr=next(r for r in color['records'] if r['case_id']==case and r['policy']==policy)
        ax.imshow(over[y0:y1,x0:x1]);ax.set_title(f"{policy}\n整页Dice={rr['wire']['dice']:.2%} / 连通={rr['connected_paths']}/{rr['total_paths']}")
    for ax in axes.flat:ax.axis('off')
    fig.suptitle('同一原图背景：绿TP / 蓝FP / 红FN；输入颜色变化，空间坐标不变')
    fig.savefig(output/'0093_color_probe.png',facecolor='white');plt.close(fig)
    source={(r['split'],r['arm']):r for r in old['summaries']}
    c=precise['summary']['control__precise_cumulative__forward'];f=precise['summary']['negative_focus__precise_cumulative__forward']
    recovery=100*(c['wire']['dice']-source[('internal_calibration10','control')]['wire']['dice'])
    blue_review=output/'blue_label_review';train_review=read(blue_review/'0025_review.json');cal_review=read(blue_review/'0093_review.json')
    route_candidate=read(blue_review/'0025_candidate/candidate.json')
    lines=['# 继续诊断 v5：源标注缺线、统计稳定性与彩色细线响应','',
        '## 1. 一句话结论','',
        '**新增的直接数据证据：训练图0025把一段真实KEY4导线的90/90像素当作有效背景，而同图KEY1的90/90像素标为导线；0093另有140个可见蓝色导线像素不在GT或源路径中。** 这类监督冲突会惩罚正确检出的导线，优先审查标注完整性，再训练拓扑目标。它不证明整个数据集的噪声率或79.46%的上限，也未测出修正后的重训收益。','',
        f"**已修复BN重估的累加顺序数值误差；没有解决整网连通瓶颈，也没有获得新的生产模型。** 同一control权重重估缓冲量，内部cal Dice从71.4754%到{c['wire']['dice']:.4%}（{recovery:+.4f}点）；单纯倒转batch只造成−0.4989点，未满足预设“大幅波动”门槛。因此不能把第13/15轮的下降直接归因于batch顺序。",
        '','FP64统计后，两组正序/倒序全部状态tensor和预测逐位一致，8,128,472个内部cal像素中的掩码变化均为0，严格预注册容差通过。这个结果只证明本轮固定batch集合的估计稳定；训练中权重演变、不同batch组合/裁块分布、未知工程关联仍未排除。','',
        f"同一FP64统计方法下，难负样本组相对control Dice={100*(f['wire']['dice']-c['wire']['dice']):+.4f}点，Recall={100*(f['wire']['recall']-c['wire']['recall']):+.4f}点，连通={f['connected_paths']-c['connected_paths']:+d}条。误检与真实导线召回仍存在取舍，不能只凭Dice提高判断整网能力。",'',
        f"原v1的79.4613%仍是当前默认结果。前轮原v1重估BN为79.3944%、连通−9条：{link(ROOT/'runs/unet_v1_BN_diagnostic_v4/report.md')}。本轮不重新打开或评分旧26页，也不替换权重。",'',
        '## 2. 实验史与固定条件','',
        f"完整原始审计：{link(OUT/'report.md')}；前轮配对训练：{link(ROOT/'runs/unet_negative_sampling_pilot_v4/report.md')}。以下均使用原93训练页/10内部cal、固定16轮checkpoint，不进行SGD，不修改标签。",'',
        '| 比较 | 唯一变化 | 保持相同 |', '|---|---|---|',
        '| EMA正序/倒序 | batch处理顺序 | 同一模型权重、925个裁块、463个batch及各batch成员、AMP、mom0.1 |',
        '| FP32累计正序/倒序 | batch处理顺序 | 同上；mom=None、原生FP32运行统计更新 |',
        '| FP64累计正序/倒序 | batch处理顺序 | 同上；原生每batch统计量仅用FP64累加 |',
        '| FP32/FP64累计 | 统计量累加精度 | 相同batch顺序、相同native输入/激活精度、相同权重 |',
        '| RGB探针 | 全图通道变换 | FP64固定统计的同一negative_focus模型、空间位置、GT、阈值 |','',
        '倒序只反转完整batch列表，末尾singleton作为完整batch移动，组内成员从未改动。各模型使用自己原定epoch16的925次采样；两模型之间训练采样不同，不能把跨模型差值当作单纯BN差值。第13/15轮checkpoint没有保存，不能精确重放其参数与缓冲量轨迹。','',
        '累计估计是463个batch的原生均值/无偏方差的等权平均，包含白padding；它不是全数据总体方差。FP64只改变短统计向量的累加精度，不改变图像、激活或可学习参数精度。原26页文件仅随旧保护列表做SHA完整性检查，没有新增图像解码、统计估计、推理评分或选参。','',
        f"协议在评分前保存：{link(output/'protocol.json')}、{link(output/'precise_extension_protocol.json')}、{link(colorpath/'protocol.json')}。数值均为原分辨率512/384全页micro@0.5，连通是2px参考管道内≥10px路径的8邻域连通，未补桥/闭运算。",'',
        '## 3. 度量与完整数字','',
        '| 组 | 统计方式 | 顺序 | Dice | P | R | 参考连通 | pin F1@0.8 |','|---|---|---|---:|---:|---:|---|---:|']
    for data in (order,precise):
        for name,r in data['summary'].items():
            arm,method,position=name.split('__');w=r['wire']
            lines.append(f"| {arm} | {method} | {position} | {w['dice']:.4%} | {w['precision']:.4%} | {w['recall']:.4%} | {r['connected_paths']}/{r['total_paths']} | {r['pin']['dice']:.4%} |")
    lines += ['',f"![BN顺序对照]({str(output/'BN_order.png').replace(chr(92),'/')})",'',
        '## 4. 错误结构与数值稳定性','',
        '| 组 / 方法 | 倒序−正序 ΔDice点 | Δ连通 | 掩码变化像素 | 最大概率差 | 原严格容差通过 |','|---|---:|---:|---:|---:|---|']
    for r in allpairs:
        stable=r.get('stable_under_original_preregistered_tolerance',r.get('stable_under_preregistered_tolerance'))
        lines.append(f"| {r['arm']} / {r['method']} | {r['delta_dice_pp']:+.8f} | {r['delta_connected_paths']:+d} | {r['changed_mask_pixels']} | {r['max_abs_probability_difference']:.9g} | {'是' if stable else '否'} |")
    lines += ['', '原门槛：最大概率差≤1e-4、掩码差异比例≤1e-5、连通差≤1。**FP32累计第一次未通过，未放宽门槛。** 仅0/4个掩码像素改变也不能说完全复现；FP64修复后才通过。相对原FP32正序输出，FP64仅改变control的1个、focus的3个掩码像素，Dice分别+0.000233/+0.000247点，精度修复本身没有可观的识别提升。','',
        'BN的momentum更新公式与momentum=None累计平均见[PyTorch BatchNorm2d官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)。由mom0.1公式推得最后20次更新权重为1−0.9^20=87.8423%；这个数学事实不等于本实验已证明尾部顺序造成原训练巨幅下降。','',
        '## 5. 根因证据与置信度','',
        '| 项目 | 证据 | 结论 / 置信度 | 最小后续证伪 |','|---|---|---|---|',
        f"| 固定checkpoint统计量影响 | 不改学习参数，control内部cal +{recovery:.4f}点 | 影响显著，高；为何旧buffer与重估不同仍未确定 | 下一轮固定保存各轮权重和buffer，再重放统计更新 |",
        '| 原batch尾部顺序是大跌主因 | EMA倒序control −0.4989点、−5条；focus −0.0587点、+2条 | 本固定权重实验不支持≥1点/≥10条门槛；训练过程轨迹未知 | 保存epoch13/15状态及每步统计/AMP scaler轨迹 |',
        '| FP32累计误差 | 概率最大差0.000590/0.001475；FP64状态及预测差0 | 本实验确认数值累加误差可消除，高 | 同batch重排多种顺序与其它checkpoint；本轮仅正/倒两序 |',
        '| RGB通道敏感 | cycle_GBR/BRG导致整cal Dice −11.91/−15.93点 | 固定模型敏感，高；不能唯一归因于导线颜色 | 将原图、符号和文字的颜色影响单独分离；不能直接宣称色彩增强重训有效 |',
        '| 简单灰度化可修复彩色细线 | 全cal Dice −6.7984点、连通−111 | 现有GT口径下固定推理方案无全局收益；GT缺线限制真实质量解释 | 如需研究灰度训练，须先核查标签完整性；本轮不采用 |',
        '| 源标注缺少真实路由 | 0025 KEY4=0/90 GT、valid=90/90；0093=0/140 GT；加入全部deleted源路径仍0覆盖 | 局部高置信数据问题；无法外推噪声率/天花板 | 训练侧完整复核、修订或标记未知监督；独立重标样本尚缺 |','',
        '第13/15轮大跌、原79.46%瓶颈、0093的全部漏检都没有被同一个原因解释。原v1的BN诊断没有整体收益；本轮新control统计重估有收益，二者不混为一谈。','',
        '## 6. 颜色推理探针日志','',
        '通道循环GBR/BRG逐像素保持算术RGB均值完全相同；mean_gray四舍五入后均值差≤1/3灰阶。没有resize、平移、重画导线或用GT变换图像。所有彩色文字、符号和填充也同时变化，因此不能唯一定位“导线颜色机制”；算术均值不是感知亮度保证。','',
        '| 输入策略 | Dice | P | R | 参考连通 | ΔDice点 |','|---|---:|---:|---:|---|---:|']
    for policy,r in color['summary'].items():
        w=r['wire'];lines.append(f"| {policy} | {w['dice']:.4%} | {w['precision']:.4%} | {w['recall']:.4%} | {r['connected_paths']}/{r['total_paths']} | {color['comparisons'][policy]['delta_dice_pp']:+.4f} |")
    style=read(colorpath/'style_audit.json')
    lines += ['', '**下表沿用冻结GT统计，不能将其FP全部理解为真实误检：视觉复核已确认部分蓝色路由缺标。** 原图像素分桶固定：chroma=max(R,G,B)−min(R,G,B)≥32且min<224为彩色墨迹；其余min<224为灰黑墨迹，其余为light。它们是图像像素代理，不是独立认证的导线颜色/线宽/图纸来源分类。','',
        '| 数据 | GT像素 | 彩色墨迹GT | 灰黑墨迹GT | light GT |','|---|---:|---:|---:|---:|']
    for split,r in style['summaries'].items():lines.append(f"| {split} | {r['GT_pixels']} | {r['chromatic_GT_pixels']} ({r['chromatic_GT_pixels']/r['GT_pixels']:.2%}) | {r['achromatic_ink_GT_pixels']} | {r['light_GT_pixels']} |")
    lines += ['', '固定原图像素桶上的彩色墨迹Recall：native77.31%，mean_gray36.86%，GBR11.70%，BRG31.31%；灰黑桶native91.30%、mean_gray91.19%。原有色彩信息在这套冻结模型中有作用，但这不证明所有蓝色细线已学好。','',
        '0093局部示例：native整页Dice26.17%、连通3/123；mean_gray整页Dice29.37%、Recall从21.38%下降至20.43%、连通仍3/123。GBR连通9/123但整页Dice24.50%。局部或单项变好仍不能采用全局变换。原v1训练过0093，曾得84.39%，只能作已见训练样本参照，不能当独立泛化证据。**图中的蓝FP表示被现有GT计为误检；其中部分实际是缺标的真实导线。**','',
        f"![0093颜色探针]({str(output/'0093_color_probe.png').replace(chr(92),'/')})",'',
        '这个颜色试验不能解释0036的灰度线漏检：R=G=B的像素在上述通道循环/算术灰度变换下保持不变。0036此前X1的45/61覆盖、两条完整参考导线0/2连通，仍说明部分像素恢复不足以输出完整网络。','',
        '## 7. 三条后续行动与本轮落地代码','',
        f"1. **优先核查训练源路由完整性。** 已新增训练0025的局部候选{link(blue_review/'0025_candidate/candidate.json')}，只补原图核实的KEY4自由线段，增加{route_candidate['added_pixels']}个3px监督像素、pin监督完全不变，未应用到冻结manifest或开始训练。现有蓝色长线扫描队列仅用于人工筛查，不能把符号/边框/文字自动改成导线。命令：`scripts/prepare_missing_route_candidate_v5.py`。全页和其它路由仍未认证；预期重训收益未知。",
        f"2. **固定checkpoint与统计配套。** 新增{link(ROOT/'src/pcb_parser/unet_bn_precise_statistics.py')}的`reestimate_bn_buffers_precise`，只接收训练RGB、关闭其它层随机行为、保持可学习权重、恢复mode/momentum，异常时恢复原buffer。下轮在固定终点使用声明过的同一统计规则，同时保留raw结果；代价本次完整93页统计约十秒量级，实测见estimation数组。改善识别效果未知，不能据cal选统计方案。",
        '3. **标注规则稳定后再检验拓扑目标与负类覆盖。** 当前参考路由并不完整，直接训练clDice也会惩罚缺标导线；先核查/修订训练侧真实连线或设置已确认的未知监督区域，再形成原损失/拓扑约束的单变量配对。统计、seed与采样同时保持固定，检查Recall、连通与错误短接。现有focus只有4页8种不同裁块，图框/文字覆盖需单独扩展，不能把cal或旧26拿来训练。收益未知；本轮新增SGD为0。','',
        '新增统计接口的5项测试和全部U-Net相关32项测试通过，包括批次成员保持、独立均值/抵消数值oracle、可学习权重/梯度不变、模式恢复、异常回滚。默认模型、原标签、旧评测脚本均保持原样。','',
        '## 8. 复现与完整性','',
        f"数值：{link(output/'evaluation.json','BN顺序诊断')}、{link(output/'precise_evaluation.json','精度修复诊断')}、{link(colorpath/'evaluation.json','RGB诊断')}；原epoch16的完整925次采样表：{link(output/'training_batch_schedules.json')}。协议和script/module SHA用于固定版本；各预测cache绑定图像和权重hash。旧679项保护文件和本轮全部训练输入hash通过。没有旧26页新评分、没有任何梯度更新、没有新的生产权重。",'',
        '已完成结果不默默覆盖。重跑需显式配置新输出目录、先写相同预注册协议。现有目录的实际命令：','',
        '```powershell', "Set-Location -LiteralPath 'C:\\Users\\Pc\\Desktop\\vision\\pcb_parser_competition'",
        '& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\diagnose_unet_BN_order_v5.py',
        '& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\verify_unet_BN_precise_v5.py',
        '& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_color_response_v5.py',
        '& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_blue_wire_label_coverage_v5.py',
        '& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\prepare_missing_route_candidate_v5.py',
        '& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\report_unet_diagnostics_v5.py','```','',
        f"本轮GPU/推理诊断脚本总墙钟耗时{(order['elapsed_seconds']+precise['elapsed_seconds']+color['elapsed_seconds'])/60:.2f}分钟；新增SGD训练0分钟。测试与报告另计。未知工程分组依赖未解决，cal10抽样排除了前轮人工复核最差10张训练图，依然不是无偏新test。"]
    missing_section=['### 两个页面的缺标证据与评测限制','',
        '训练0025的同图对照：','', '| 线段 | 原图RGB | GT导线像素 | 包含全部deleted路径后 | 当前训练valid | focus16检测 |','|---|---|---:|---:|---:|---:|']
    for r in train_review['checks']:
        missing_section.append(f"| {r['label']} y={r['y']}，x50..139 | {r['RGB_unique']} | {r['current_GT_positives']}/90 | {r['all_states_source_path_positives']}/90 | {r['actual_training_valid_pixels']}/90 | {r['predictions']['negative_focus']['detected_pixels']}/90 |")
    missing_section += ['', 'KEY4核心像素RGB=(0,0,128)，算术均值42.67；KEY1=(18,18,136)，均值57.33。缺标那段更深，不能解释成它颜色过淡。KEY4不在源keyPointPaths里，加入deleted路径也没有这段，源skeleton同样没有。当前Dataset直接给这些像素target=0、valid=1；独立BCE数值oracle验证logit=0、pos_weight8、y=0时梯度为+0.5，梯度下降会压低这类真实导线的预测。这里仅证明局部监督冲突，不把它解释为全部漏检原因。','',
        f"核查记录：{link(blue_review/'0025_review.json')}。训练侧局部候选（未应用）如下，橙色仅为补标候选，不是模型输出：",'',
        f"![0025局部候选]({str(blue_review/'0025_candidate/comparison.png').replace(chr(92),'/')})",'',
        '0093是当前10页内部cal之一：两条已目视核实的自由线段，y448/y546、x110..179，合计140个连续RGB=(0,0,128)像素，GT=0/140。全部85条源边（81active、4deleted、0flywire）纳入重建后仍0/140，源skeleton也0/140，定位到源几何缺失，非当前mask栅格化或flywire/deleted过滤造成。该页没有进入本轮93训练页；旧v1训练过它。','',
        '这140个真实线段像素，native检测94个，mean_gray检测58个，全部按旧GT计作FP。同一次灰度变换，旧GT整页Dice从26.17%升至29.37%，实际已核实线段覆盖却从94/140降至58/140。因此局部Dice变高不充分证明实际连线检测变好。没有修订cal GT、没有删除0093、没有计算更换GT后的“新整体Dice”、没有估计标注上限。','',
        f"核查记录：{link(blue_review/'0093_review.json')}。这些都是代理局部目视核查，不是独立第二标注。",'',
        f"训练93页扫描到10页550条蓝色长水平stroke候选、共33968个候选像素；cal10有2页17条、1524像素。扫描阈值≥40px且>90%距GT超过2px，包含符号/边框/文字和厚线重复行，**这些数不能称真实漏标数量、数据噪声率或需要补标的导线长度**。队列：{link(blue_review/'candidate_scan.json')}。",'']
    report='\n'.join(lines)+'\n'
    report=report.replace('## 7. 三条后续行动与本轮落地代码','\n'.join(missing_section)+'\n## 7. 三条后续行动与本轮落地代码')
    (output/'report.md').write_text(report,encoding='utf-8')
    paths=re.findall(r'\]\((C:/[^)]+)\)',report);assert all(Path(p).exists() for p in paths)
    assert len(re.findall(r'^## \d+\.',report,re.M))==8
    protected=read(ROOT/'runs/unet_negative_sampling_pilot_v4/protected_inputs.json')|read(ROOT/'runs/unet_negative_sampling_pilot_v4/additional_source_attestation.json')
    assert all(sha(p)==v for p,v in protected.items())
    write(output/'final_checks.json',dict(status='complete',protected_original_files=assert_protected(),frozen_training_inputs_passed=len(protected),report_links_passed=len(paths),
        new_SGD_updates=0,old26_new_scoring=False,all_precise_state_and_predictions_equal=all(r['all_state_tensors_bitwise_equal'] and r['max_abs_probability_difference']==0 for r in precise['pairs']),
        original_tolerance_unchanged=True,original_tolerance_passed=all(r['stable_under_original_preregistered_tolerance'] for r in precise['pairs']),
        unet_tests_passed=32,new_bn_tests_passed=5,production_sha256=sha(ROOT/'weights/unet_wire_pin_v1.pt'),production_changed=False,
        missing_route_candidate_applied=False,cal_labels_changed=False,local_label_audits=['0025_train','0093_calibration'],
        current_torch_version=torch.__version__,new_files_sha256={str(ROOT/path):sha(ROOT/path) for path in ('src/pcb_parser/unet_bn_statistics.py','src/pcb_parser/unet_bn_precise_statistics.py','scripts/diagnose_unet_BN_order_v5.py','scripts/verify_unet_BN_precise_v5.py','scripts/audit_unet_color_response_v5.py','scripts/report_unet_diagnostics_v5.py','scripts/audit_blue_wire_label_coverage_v5.py','scripts/prepare_missing_route_candidate_v5.py')}))
    print(json.dumps(dict(report=str(output/'report.md'),tests=32,all_precise_predictions_equal=True,old_sources=assert_protected(),new_training=0),ensure_ascii=False))


if __name__=='__main__':main()
