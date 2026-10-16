"""Render the finished paired pilot, with causal limits and reproducible evidence."""
import json
from unet_audit_common import ROOT, OUT, read, sha, write, assert_protected


def link(path,label=None):
    return f"[{label or path.name}]({str(path).replace(chr(92),'/')})"


def write_report(run,cfg,result):
    review=ROOT/"runs/unet_training_label_review_v4"
    labels=read(review/"summary.json");distribution=read(review/"actual_sampling_distribution.json")
    training=read(run/"summary.json")
    score={(r["split"],r["arm"]):r for r in result["summaries"]}
    internal=next(r for r in result["comparisons"] if r["split"]=="internal_calibration10")
    external=next(r for r in result["comparisons"] if r["split"]=="frozen_external26")
    bn=[r for r in result["bn_comparisons"] if r["split"]=="internal_calibration10"]
    common_internal=next(r for r in bn if r["contrast"]=="sampling_with_common_BN_method")
    common_external=next(r for r in result["bn_comparisons"] if r["split"]=="frozen_external26" and r["contrast"]=="sampling_with_common_BN_method")
    passed=result["preregistered_internal_success"]
    lines=["# U-Net改进验证：局部标签修正与难负样本配对试验 v4", "",
        "## 1. 结论", "",
        f"预先规定的采样成功门槛：**{'通过' if passed else '未通过'}**。固定第16轮，处理组相对对照组在10张内部校准页的Dice变化为 **{internal['delta_dice_pp']:+.4f}个百分点**，Recall变化{internal['delta_recall_pp']:+.4f}点，连通路径变化{internal['delta_connected_paths']:+d}条；原26页冻结验证的Dice变化{external['delta_dice_pp']:+.4f}点。",
        f"决定：`{result['decision']}`。{'可保留为后续确认试验候选，仍需更长训练和多seed。' if passed else '本轮25%/16轮/单seed配方不支持采用；不能据此否定所有难负样本方案。'}默认v1权重未替换。", "",
        "这轮回答的是配对采样方案是否有效。两组共同使用局部修订标签，没有未修订标签的平行对照，因此**没有测出标注修正本身带来多少提升**。BN诊断只改变训练图估计的运行统计量，单独列示，不能把组合差值全部归因于采样。", "",
        f"**实际采用仍不成立。** 难负样本组旧26页Dice={score[('frozen_external26','negative_focus')]['wire']['dice']:.4%}，低于原v1的79.4613%；两组采用同一训练图BN重估方法后，内部cal ΔDice={common_internal['delta_dice_pp']:+.4f}点，但Recall={common_internal['delta_recall_pp']:+.4f}点、连通={common_internal['delta_connected_paths']:+d}条；旧26页ΔDice={common_external['delta_dice_pp']:+.4f}点，同时Recall={common_external['delta_recall_pp']:+.4f}点、连通={common_external['delta_connected_paths']:+d}条。不能只看Dice宣称整网能力提高。raw内部差值的页bootstrap95%区间也跨0。", "",
        f"此前79.4613%的根因审计见{link(OUT/'report.md','完整审计报告')}。其中GT外2像素以上的FP占全部误差57.84%；局部位置相位的改善未转化为26页整体提升。因此本轮从已核对的非导线误检入手。", "",
        "## 2. 数据与标签核查", "",
        "原103张训练页按整页划为93训练、10内部校准；确切同图组保留在同一侧。校准页从未复核的单例图中抽取，排除了用于标签审查的最差10张训练图，**这会使内部集偏向未被挑出的图，不能称无偏随机独立测试**。两组从相同随机权重开始，避免用已学过这10页的v1预训练权重。工程分组ID仍缺失，无法排除工程级相关。原26页验证标签冻结；这是旧验证集，不是新获得的独立test。", "",
        "同一源标注的两个几何字段：2999条边可比较，568070个路径中心采样点中76233个（13.42%）距源skeleton超过2像素；1123条边存在这种差异。这些是同一标注内部的字段分歧，不是独立标注一致性、真实噪声率或Dice上限。0129/0215有路径字段新增的空白路线，0088/0254两个字段也可能共同错位，不能自动信任任一字段。", "",
        "两组共用既有0073/0088派生修订，并加入以下4页的局部候选。原始annotation、v1/v2标签与默认权重均保持原样。", "",
        "| 训练页 | 局部候选 | 新增wire像素 | 删除wire像素 | tip位置变更 |", "|---|---|---:|---:|---:|"]
    reasons={"0129":"BUZZER供电及二极管支路按PNG重新走线", "0215":"删除E_5末6段空白/穿元件路径", "0254":"上方及电容支路向下移10–11像素", "0265":"Vs电容支路与接地引线按PNG调整"}
    for r in labels["candidates"]:
        lines.append(f"| {link(review/'candidates'/r['case_id']/'comparison.png',r['case_id'])} | {reasons[r['case_id']]} | {r['added_pixels']} | {r['removed_pixels']} | {r['candidate_pin_points_changed']} |")
    lines += ["", "4页合计新增1846、删除3315个wire像素，5个tip位置变更。改动范围与图像尺寸、wire/pin差分均有断言；重建路径栅格必须逐像素等于最终监督标签。它们仍是代理局部复核草稿，没有独立第二标注或整页认证。0265源图中部分引脚的图结构归属仍不可靠，候选仅用于像素监督，不作为已确认整网真值。", "",
        "确认12个无导线ROI，其中v1误检的9个共2402个FP，才进入focus池。实际focus类别仅IC/模块框和Audio Source框；3个原本没有FP的表格/文字ROI没有进入池，不能声称已经训练了完整图框/文字负类。9个ROI折成8个不同512裁块，覆盖4张训练图。", "",
        "## 3. 固定试验设计与实际采样", "",
        "| 参数 | 固定值 |", "|---|---|",
        f"| seed / 初始化 | {cfg['seed']} / 相同随机初始权重；非v1 warm start |",
        "| 模型 | WirePinUNet，RGB、双头、base32、原18层BN |",
        f"| epoch / batch / optimizer | {cfg['epochs']} / {cfg['batch_size']} / AdamW，lr={cfg['learning_rate']}，wd={cfg['weight_decay']} |",
        "| wire / pin损失 | 原BCE(pos8)+Dice，0.5×pin BCE(pos20)，valid mask；AMP |",
        "| 图像 / 裁块 | 原RGB无缩放无增强；512×512，stride384，白padding |",
        "| 每轮采样 | 两组各925次；处理组替换231次=24.97%，其余槽位逐项一致 |",
        "| checkpoint / threshold | 固定last16；wire=0.5；pin=0.8、quadratic、匹配半径5px |",
        "| 成功门槛 | 内部cal micro Dice≥+1点，Recall和参考路径连通不下降，已核对训练负类ROI FP下降 |", "",
        f"完整配置：{link(ROOT/'configs/unet_negative_sampling_pilot_v4.json')}；运行配置/初始权重hash：{link(run/'configuration.json')}。", "",
        "| 第1轮实际曝光（重叠裁块重复计数） | control | negative_focus |", "|---|---:|---:|"]
    d={r["arm"]:r for r in distribution["records"]}
    for key,label,fmt in (("valid_pixels","有效像素","d"),("wire_positive_pixels","wire正像素","d"),("positive_fraction","wire正例占比",".4%"),("empty_GT_tiles","空GT裁块","d")):
        lines.append(f"| {label} | {format(d['control'][key],fmt)} | {format(d['negative_focus'][key],fmt)} |")
    lines += ["", "采样不仅改变负类出现次数，也改变图样、上下文和裁块原点相位分布。positive占比从1.3238%降至1.2077%，空GT裁块反而从212减至153，因为focus裁块仍含其它真实导线。没有把整个ROI附近裁块强制标成背景。采样差值只能归因于这一完整干预，不能唯一归因于负类语义。", "",
        f"实际配对训练耗时{training['elapsed_seconds']/60:.2f}分钟；最初9–11分钟/臂的估时不准确，配置保持原定16轮。运行中校准波动并未用于选轮或提前停止。\n\n![训练与内部校准曲线]({str(run/'curves.png').replace(chr(92),'/')})", "",
        "## 4. 固定终点评价", "",
        "全部是恢复原图坐标后的全页micro，而不是训练日志的tile累计Dice。BN表项为相同last16卷积权重的统计量诊断。连通指标：现有参考路径≥10像素，在2像素参考管道内，预测8邻域分量须覆盖全部路径顶点；未做闭运算、补桥或GT修复预测。", "",
        "| 数据 | checkpoint | Dice@0.5 | P | R | TP / FP / FN | 连通路径 | pin F1 |", "|---|---|---:|---:|---:|---|---|---:|"]
    for r in result["summaries"]:
        w=r["wire"];lines.append(f"| {r['split']} | {r['arm']} | {w['dice']:.4%} | {w['precision']:.4%} | {w['recall']:.4%} | {w['tp']} / {w['fp']} / {w['fn']} | {r['connected_paths']}/{r['total_paths']} | {r['pin']['dice']:.4%} |")
    lines += ["", "原v1在26页Dice=79.4613%，连通1150/1548。它用103训练页、不同seed、60轮并按旧val loss选57轮；本轮93页、修订标签和16轮随机初始化。可以作现状参照，**不能把新模型与v1的差值解释为单一采样或标注修正收益**。", "",
        "## 5. 配对差值与BN诊断", "",
        "| 比较 | 数据 | ΔDice点 | ΔP点 | ΔR点 | Δ连通条数 | Δ负类ROI FP | 配对页bootstrap95% ΔDice点 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in result["comparisons"]+result["bn_comparisons"]:
        ci=r["paired_page_bootstrap95_delta_pp"]
        lines.append(f"| {r['contrast']} | {r['split']} | {r['delta_dice_pp']:+.4f} | {r['delta_precision_pp']:+.4f} | {r['delta_recall_pp']:+.4f} | {r['delta_connected_paths']:+d} | {r['negative_ROI_FP_delta']:+d} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ["", "主成功门槛仅看raw采样配对，未用BN结果追溯改变门槛。Bootstrap为10000次相同页配对重采样；页数少、未知工程依赖未校正，单次训练seed也未覆盖训练方差。旧26页上的结果与阈值曲线仅诊断，没有用于选参数、选checkpoint或上线。", "",
        "BN方案在评分前写入协议：同93张训练RGB，925个原裁块，batch8，shuffle seed20261014，18个BN清空统计并累计平均一遍；其它层eval，无梯度更新；没有读取cal/旧26图来估计统计量。hash确认全部卷积和BN可学习参数未变，只改变54个running mean/var/count buffers。不同BN输入分布本身也是一个干预；这不证明GroupNorm重训收益，更不单独证明原v1的79.46瓶颈就是BN。", ""]
    for r in bn:
        lines.append(f"- {r['contrast']}：内部cal ΔDice={r['delta_dice_pp']:+.4f}点、ΔRecall={r['delta_recall_pp']:+.4f}点、Δ连通={r['delta_connected_paths']:+d}条。")
    lines += ["", f"BN协议：{link(run/'BN_diagnostic_protocol.json')}；逐buffer变更与权重hash：{link(run/'BN_diagnostic_summary.json')}。", "",
        "## 6. 具体错误与可视证据", "",
        "局部图相同缩放、相同阈值：绿TP、蓝FP、红FN，均相对尚未整页认证的现有标签。原图和GT同时显示，防止把GT错误当模型错误。页面P/R为整页值，不能当局部ROI指标。", ""]
    for case,desc in (("0281","训练页IC边框：用来核查负类误检是否实际减少。"),("0170","旧验证页图框/标志：该风格没有加入focus池，检验是否泛化。"),("0036","旧验证页X1细线：观察误检减少时是否同时损失真实导线。")):
        lines += [f"{case}：{desc}", "", f"![{case}对照]({str(run/f'comparison_{case}.png').replace(chr(92),'/')})", ""]
    lines += ["固定先前审计的0036线段采样（半开x区间，未重新挑位置；像素覆盖不等于完整网络连通）：", "",
        "| 线段 | checkpoint | ≥0.5像素 / 长度 | p中位数 |", "|---|---|---|---:|"]
    for r in result["fixed_prior_audit_probes"]:
        lines.append(f"| {r['probe']} | {r['arm']} | {r['detected_pixels']}/{r['pixels']} | {r['median_probability']:.6f} |")
    local_path=run/"local_path_checks.json"
    if local_path.exists():
        checks=read(local_path)
        lines += ["", "X1的完整局部参考导线也独立检查：E_1=(601,80)→(691,80)，90px；E_8=(722,80)→(768,80)，46px。", "",
            "| checkpoint | 连通参考导线 |", "|---|---|"]
        for name,r in checks["models"].items():lines.append(f"| {name} | {sum(p['connected_within_reference_tube'] for p in r)}/{len(r)} |")
        lines += ["",f"X1的45/61覆盖只是部分恢复，两段仍断裂。逐路径记录：{link(local_path)}。"]
    lines += ["", "训练负类ROI总FP属于已经参与采样的训练图表现，不能当泛化证明；真正的采样泛化判断仍结合内部cal的Recall和参考连通。没有独立引脚/整网真值，参考路径指标不能认证整网正确率。", "",
        "## 7. 当前不足与后续优先级", ""]
    if not passed:
        lines += ["本轮配方未满足预注册门槛，停止采用或延长此配方的自动决定。固定16轮可能尚未收敛，单seed不能区分稳定失败与训练方差；同样不能凭挑选较好epoch宣布成功。"]
    else:
        lines += ["配对内部门槛通过，但仍只有单seed、16轮和偏向未复核图的内部cal。应保留候选，后续确认时预定更长轮数/多seed，再看Recall、连通和新负类风格是否保持。"]
    gains=[r for r in bn if r["contrast"] in ("control_BN_only","focus_BN_only")]
    if any(r["delta_dice_pp"]>=1 for r in gains):
        lines += ["BN运行统计的影响在本轮可以直接干预验证，至少一组内部cal提升≥1点。下一轮采样/拓扑对比前应先固定统一的归一化与统计估计协议，否则单个epoch的校准波动可能淹没干预效应。这里不据旧26结果选择或替换权重。"]
    else:
        lines += ["BN重估未显示一致明显的内部cal恢复；归一化仍可作为后续单变量消融，不能把训练波动全部归因于BN。"]
    lines += ["", "1. **优先补标注规则及独立复核样本。** 明确引脚露出导线、元件内部线、交点/非交叉点、虚线/粗线和图框的规则；4页局部草稿不等于整个训练集可认证。效果只在固定独立重标样本上检验，不能用移动GT计算上限。",
        "2. **先稳定训练，再确认采样。** 预定seed与轮数，固定统计协议；扩大已核对的负类风格覆盖及不同裁块原点，保留Recall/连通门槛。采样率或新归一化必须各自形成配对试验，避免叠加后归因。",
        "3. **另做拓扑目标单变量实验。** 标签与训练稳定后加入拓扑约束，与原损失同seed同数据比较；同时报告pixel Dice、细线覆盖、参考连通和错误短接。像素提高不保证完整网输出。", "",
        "## 8. 复现与完整性", "",
        f"证据：{link(run/'evaluation.json','全页记录与配对汇总')}、{link(review/'actual_sampling_distribution.json','实际曝光分布')}、{link(review/'report.md','标签审查与修正准备报告')}。所有旧源文件的679项hash保持一致，当前训练输入、初始权重和新候选另有冻结hash；CUDA deterministic配置仍不能承诺所有算子逐位完全复现。", "",
        "训练前，采样额度/复现性/非替换槽位一致、focus裁块保留真实标签及拒绝越界数据的3项新增测试通过；全部U-Net相关测试27项通过。终点评价另检查129页参考路径栅格与监督逐像素一致、所有权重/图像hash、报告链接和局部完整路径。", "",
        "复跑训练需先在新输出目录配置对应路径；脚本拒绝覆盖现有训练。下面是本轮执行顺序，评价缓存绑定图像/checkpoint hash：", "", "```powershell",
        "Set-Location -LiteralPath 'C:\\Users\\Pc\\Desktop\\vision\\pcb_parser_competition'",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\prepare_unet_training_review_v4.py",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\describe_unet_sampling_v4.py",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\train_unet_negative_sampling_v4.py --allow-agent-draft --check-only",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\train_unet_negative_sampling_v4.py --allow-agent-draft",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\reestimate_unet_BN_v4.py",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\evaluate_unet_negative_sampling_v4.py",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\diagnose_original_unet_BN_v4.py",
        "& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\finalize_unet_pilot_v4.py", "```", "",
        f"训练{training['elapsed_seconds']/60:.2f}分钟，BN统计重估{result['BN_diagnostic']['elapsed_seconds']/60:.2f}分钟，全页评价{result['elapsed_seconds']/60:.2f}分钟。" ]
    (run/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    original_path=ROOT/"runs/unet_v1_BN_diagnostic_v4/evaluation.json"
    if original_path.exists():
        direct=read(original_path);delta=direct["delta"]
        block=["### 原v1的直接BN干预", "",
            "额外诊断也在评分前冻结协议：同一epoch57，只用原103训练页RGB重估统计，986裁块、batch8、seed20261014；54个buffer改变，全部可学习权重hash相同。旧26页不参与估计或参数选择。", "",
            "| 原模型 | Dice@0.5 | P | R | 连通路径 |", "|---|---:|---:|---:|---|"]
        for name,r in direct["summary"].items():block.append(f"| {name} | {r['wire']['dice']:.4%} | {r['wire']['precision']:.4%} | {r['wire']['recall']:.4%} | {r['connected_paths']}/{r['total_paths']} |")
        block += ["",f"ΔDice={delta['Dice_pp']:+.4f}点、ΔRecall={delta['R_pp']:+.4f}点、Δ连通={delta['connected_paths']:+d}条，页bootstrap95% ΔDice={delta['paired_page_bootstrap95_Dice_pp']}。这套重估方案未改善原v1的Dice或连通，**不能把原79.46%瓶颈归因于BN统计失准**。本轮新control的BN不稳定与原v1的瓶颈应分别判断。", "",
            f"详情与复现：{link(original_path.parent/'report.md','原v1 BN诊断')}。", ""]
        text=(run/"report.md").read_text(encoding="utf-8")
        (run/"report.md").write_text(text.replace("## 6. 具体错误与可视证据","\n".join(block)+"\n## 6. 具体错误与可视证据"),encoding="utf-8")
    script_names=["prepare_unet_training_review_v4.py","describe_unet_sampling_v4.py","train_unet_negative_sampling_v4.py","reestimate_unet_BN_v4.py","evaluate_unet_negative_sampling_v4.py","report_unet_negative_sampling_v4.py"]
    write(run/"final_checks.json",dict(status="training_BN_and_endpoint_evaluation_complete",protected_original_files=assert_protected(),
        original_production_sha256=sha(ROOT/"weights/unet_wire_pin_v1.pt"),production_changed=False,
        script_sha256={str(ROOT/"scripts"/name):sha(ROOT/"scripts"/name) for name in script_names},
        learned_weights_equal_before_after_BN=all(r["learned_weights_unchanged"] for r in result["BN_diagnostic"]["records"]),
        external_validation_used_for_parameter_or_checkpoint_selection=False,independent_test=False,partial_label_drafts=True))


if __name__=="__main__":
    run=ROOT/"runs/unet_negative_sampling_pilot_v4"
    write_report(run,read(run/"configuration.json"),read(run/"evaluation.json"))
