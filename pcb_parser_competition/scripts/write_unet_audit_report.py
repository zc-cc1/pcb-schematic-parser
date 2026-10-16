"""Build the requested eight-part report from frozen audit evidence."""
import csv
from pathlib import Path
import numpy as np
from unet_audit_common import ROOT, OUT, read, write, sha, aggregate, assert_protected


def link(path,label=None,line=None):
    path=Path(path);s=str(path).replace("\\","/")+(f":{line}" if line else "")
    return f"[{label or path.name}]({s})"


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers)]+["| "+" | ".join(map(str,r))+" |" for r in rows])+"\n"


def pct(x):return f"{100*x:.4f}%"


def main():
    b=read(OUT/"baseline_reproduction.json");m=read(OUT/"metric_audit.json");e=read(OUT/"error_decomposition.json")
    h=read(OUT/"experiment_history.json");a=read(OUT/"inference_audit.json");v=read(OUT/"manual_review30.json");f=read(OUT/"final_checks.json")
    rows_by_name={r["variant"]:r for r in a["summary"]};boot={r["variant"]:r["paired_page_bootstrap95_pp"] for r in f["bootstrap"]}
    texts=[]
    def add(s):texts.append(s.strip()+"\n")
    add("# UNet导线分割根因审计 · 2026-10-14\n\n范围：原模型、既有训练记录、103张训练页与26张去重验证页；原始图片、标签、划分、默认模型及生产参数均保持冻结。本轮新增训练次数 **0**。")
    add("## 1. 一句话结论\n\n**79.46%是对未认证、固定3px描边的草稿标签计算的整图micro Dice；当前主导误差是远离GT的FP（总误差的57.84%），其中已证实有图框/器件边/文字误检，也混有标注缺段和偏移；同时存在真实细线漏检及泛化差距，尚不能认定标注天花板或模型容量上限。**")
    add(f"基线逐像素复现成功：TP={b['micro']['tp']:,}、FP={b['micro']['fp']:,}、FN={b['micro']['fn']:,}；P={pct(b['micro']['precision'])}、R={pct(b['micro']['recall'])}、Dice={pct(b['micro']['dice'])}。固定checkpoint epoch57，SHA256 `{read(OUT/'protocol.json')['checkpoint_sha256']}`。所有{assert_protected()}个冻结文件的SHA256均未改变。")
    add(f"![审计总览]({str(OUT/'audit_summary.png').replace(chr(92),'/')})")
    add("## 2. 实验史重建：为什么不增反降")
    history=[]
    for r in h["experiments"]:
        history.append([r["id"],r["variables"],r["validation_cases"],pct(r["wire_micro"]["dice"]),r.get("seed","—"),r["epochs"],r["best_epoch"]])
    old=h["other_existing_inference_comparison"]
    history.insert(2,["历史推理v2候选","native+1.5倍均值、阈值0.35；同时调整pin出口（多个变量）",26,pct(old["all_unique"]["optimized"]["wire"]["f1_dice"]),"—",0,57])
    add(table(["实验","改变量","验证页","整图micro Dice","seed","新训练epoch","所选epoch"],history))
    add("- 79.9816%→79.4613%来自删除0268、0274两张验证重复图导致的权重变化，模型和原预测没有变化。原始28页口径仍保留。\n- control/contrast/rotation的26页比较均为相同验证标签、整图micro、阈值0.5。相对配对control，光度增强+0.4119点，90°旋转增强−2.6060点；下降不是micro/macro或阈值差异制造的。\n- v1→control同时改变训练标签（仅0073/0088局部修正）、seed、学习率/调度、微调轮数及验证去重，不能把−0.3567点全部归因于某个模块。control与增强臂才是主要消融对照；旋转臂复用既有control，未重新运行一对随机种子。\n- v1按最小验证总loss选57；微调也按原图验证最小总loss选7/10/9。loss包含pin任务；这不等于按wire Dice选最优。v1历史最高tile Dice出现在56（79.4210%），57为79.3957%，只有0.0253点差，无法解释主要瓶颈。\n- 历史推理v2候选在13页development上wire Dice87.2674%→87.6045%，另13页74.0396%→71.9183%；全部26页79.4613%→78.2368%。它未通过验收，正式wire出口仍为native/0.5；正式pin参数另行改到0.8/K5/quadratic。两组13页都曾用于模型选择/开发，均不是独立test。\n- 工作目录不是Git仓库，因此无git log。可用原训练完整60轮日志、三臂16轮记录和checkpoint均已读取；notebook清单及配置/标签/权重SHA见实验史JSON。")
    add(f"完整历史：{link(OUT/'experiment_history.json')}；人类可读表：{link(OUT/'01_experiment_history.md')}。原训练与微调曲线分别见总览及 {link(OUT/'warmstart_curves.png')}。")
    add("## 3. 复现与度量审计")
    tiles=m["overlapping_valid_tiles_from_same_stitched_predictions"]
    fit=rows_by_name["single_fit512"]
    metrics_rows=[
        ["原图整页micro / 阈值0.5",pct(b["micro"]["dice"]),"26页每个原图像素计一次"],
        ["原图macro / 阈值0.5",pct(m["baseline_macro"]),"26张图等权"],
        ["阈值扫描诊断最优",pct(m["diagnostic_best_threshold"]["micro"]["dice"]),"0.80；+0.4820点，仅诊断，不采用/选参"],
        ["排除全背景图micro",pct(m["excluding_background_images_micro"]["dice"]),"26页全有GT导线，排除数量0"],
        ["相同拼接概率图切回重叠tile：micro",pct(tiles["micro"]["dice"]),"140块，10块GT为空；重叠像素重复加权"],
        ["上述tile macro / 空GT+空预测记1",pct(tiles["macro_with_empty_perfect"]),"这是tile等权平均，不能与整图macro混用"],
        ["上述tile macro / 排除GT空块",pct(tiles["macro_without_GT_empty"]),"130块；只减少0.3644点，无明显空块抬分"],
        ["实际网络输入分辨率","无缩放差项","原分辨率512裁块/stride384/白色padding；不是整图resize512"],
        ["反事实整页fit512：恢复原图后micro",pct(fit["micro"]["dice"]),"这是新推理策略；相对默认−32.6816点"],
        ["同一fit512概率图：缩放输入网格micro",pct(fit["resized_input_grid_micro"]["dice"]),"GT仅在此诊断用NEAREST缩放；53.3497→46.7796差6.5701点，含插值/栅格/像素加权，不能都称缩放损失"],
        ["骨架exact Dice",pct(m["skeleton_micro"]["dice"]),"P65.3085%、R79.8307%；精确1px骨架位置仍敏感"],
        ["容差1px relaxed F1",pct(m["relaxed_micro"]["1"]["f1"]),"P76.9726%、R92.2161%；双向最近距离匹配"],
        ["容差2px relaxed F1",pct(m["relaxed_micro"]["2"]["f1"]),"P78.2665%、R93.5784%；不证明电气网络正确"],
        ["clDice式骨架/区域交叠F1",pct(m["topology_clDice"]["f1"]),"P75.3390%、R91.5131%；本轮没有训练clDice loss"],
    ]
    for radius,row in m["boundary_band"].items():
        metrics_rows.append([f"GT膨胀−腐蚀边界带 r={radius}",pct(row["dice"]),f"含总错误{100*row['fraction_of_total_errors']:.4f}%；GT像素包含{100*row['fraction_of_GT_inside_band']:.4f}%"])
    for offset in ("1,0","0,1","1,1","2,0","0,2","2,2"):
        # Stored keys use Python tuple syntax.
        key="("+offset.replace(",",", ")+")"
        row=m["GT_shift_sensitivity"].get(key)
        if row is None:
            key=next(k for k in m["GT_shift_sensitivity"] if k.replace(" ","").strip("()") == offset)
            row=m["GT_shift_sensitivity"][key]
        metrics_rows.append([f"GT vs 平移GT {offset}px",pct(row["dice"]),f"裁掉GT像素{row['lost_GT_at_canvas']}；敏感性，不是双标注"])
    add(table(["口径","数值","解释"],metrics_rows))
    add("### 应当纠正的两种推断\n\nGT自己与1px平移GT的Dice下降，只能说明栅格位置敏感。它不提供真实标注偏移概率，也没有第二人的标签，因此不能推出“80%接近可分辨上限”。同样，把训练过的checkpoint换一批grouped评测页，不能撤销训练时已经见过的数据；差值不能自动叫作泄露程度。")
    add("### 数据、标签与划分\n")
    stats=[]
    for name in ("train103","val28","val26_unique"):
        r=b["stats"][name];stats.append([name,r["images"],f"{r['positive_pixels']:,}/{r['pixels']:,}",pct(r["positive_fraction"]),f"W{r['width_min']}–{r['width_max']}；H{r['height_min']}–{r['height_max']}",f"{r['tiles']} / 空{r['empty_wire_tiles']}"])
    add(table(["划分","图数","正类/全像素","正类比例","原图尺寸范围(px)","512块数"],stats))
    hist=[]
    for name in ("train103","val26_unique"):
        d=b["stats"][name]["gt_width_histogram"];hist.append([name,d.get("1",0),sum(n for k,n in d.items() if 2<=int(k)<=3),sum(n for k,n in d.items() if int(k)>=4),str(d)])
    add(table(["GT骨架估算宽度","1px点数","2–3px点数","≥4px点数","详细直方图"],hist))
    add("- 300页中132页结构可用：103训练、28验证、1排除；完整人工认证页数0。现有宽度统计是GT骨架距离变换 `round(max(1,2d−1))`，验证26页83305个骨架点中81330为3px（97.6292%）；不是PNG实际导线宽度直方图。0036已实测1px线在PNG中存在，但监督统一描3px。\n- wire标签由CVJsonStd_raw中的active、非deleted、非flywire `keyPointPaths`用Pillow ImageDraw.line绘制，坐标round、fill255、width3。它是路径几何栅格草稿，未找到对应原始CAD/PDF像素真值或独立人工像素标签。标注最初是人工/半自动/自动生成无法进一步确认。\n- 全部131页wire mask只有0/255。训练/默认推理都不resize图像或mask；不存在wire mask双线性软标签bug。pin heatmap使用sigma2 Gaussian软值是另一通道的有意设计。训练有效区域屏蔽白色padding。\n- 灰度/GT对齐线索：±3px窗口平均墨迹相关峰在(0,0)的有17/26；其余9页峰分散在不同方向。预测对整体平移GT评测全部更差（例如+1x为72.78%、+1y为62.40%），不支持一个统一常量偏移解释全体误差。该墨迹方法包含文字/器件，不能替代局部语义审查。\n- 原图case先分split再切tile；同图不同patch跨split数量0。解码像素完全相同的重复组4：val0026/0268、val0029/0274、train0028/0266、train0069/0269；跨split完全相同组0，裁白边身份跨split组0。64bit pHash≤8的跨split候选0。这只排除了已检查的近重复形式。\n- 300页source/工程分组ID全部空，用户确认暂无可信ID；工程级分组评测无法计算。已存在的case级分组口径仍79.4613%、Δ0。不能把unknown写成“没有工程泄露”。重复图GT Dice=100%属于复制标签，不是标注者间一致性。\n- 26页GT覆盖纯白像素12.0780%，固定3px描边会包含白色邻域，单靠这个比例也不能判标签错误；本轮采用具体路径/原图核对。")
    add(f"![阈值曲线]({str(OUT/'dice_threshold_curve.png').replace(chr(92),'/')})\n\n0.50–0.90均在扫描最优值0.5个点以内，属于宽平台。数字来源：{link(OUT/'metric_audit.json')}。")
    add("## 4. 错误分解：具体错在哪里")
    part_rows=[]
    for key,label in (("FP_far_GT2","FP距GT>2px"),("FP_near_GT2","FP距GT≤2px"),("FN_far_prediction2","FN距预测>2px"),("FN_near_prediction2","FN距预测≤2px")):
        part_rows.append([label,f"{e['partitions'][key]:,}",pct(e["partition_error_fractions"][key])])
    add(table(["互斥几何误差分区","像素","占FP+FN"],part_rows))
    add("总误差115242像素，FP占72.6706%。2px以内FP+FN是27.9872%，是边界/微小缺口的几何代理，并非都由标注噪声造成；GT边界带r1只覆盖总误差32.3997%，r2为42.0680%且已覆盖99.6081%GT，因此不能说“误差大多在边界”。")
    bucketnames={"GTwidth_1":"GT宽1px参考邻域","GTwidth_2_3":"GT宽2–3px参考邻域","GTwidth_ge4":"GT宽≥4px参考邻域","outside_GT2px":"GT2px邻域外","image_border32":"图像边缘32px","image_interior":"其余内部","dense_GT_proxy3pct64":"64窗口GT占比≥3%","sparse_GT_proxy3pct64":"64窗口GT占比<3%","source_mode_RGB":"RGB来源模式(18页)","source_mode_RGBA":"RGBA来源模式(8页)"}
    add(table(["线宽/空间/来源代理","Dice","P","R","TP / FP / FN"],[[bucketnames[n],pct(r["dice"]),pct(r["precision"]),pct(r["recall"]) if r["tp"]+r["fn"] else "N/A",f"{r['tp']} / {r['fp']} / {r['fn']}"] for n,r in e["width_buckets"].items()]))
    add("线宽分桶按最近GT骨架宽度分配GT及其≤2px邻域，远FP另列，所有误差未被遗漏。1px桶误差5933像素，仅占总误差5.1483%；2–3px桶误差41955，占36.4060%；≥4px桶703，占0.6100%。这不能支持“1px线占全部主要Dice误差”。空间稀疏桶低分包括图框/器件和漏标，GT稀疏不是经过认证的电路稀疏。RGB/RGBA是文件模式代理，创建软件/图集未知；移除0170的只读影响分析中其余7张RGBA Dice85.1267%，所以不能据模式差异宣布RGBA风格本身难学。0170仍留在所有正式26页比较中。")
    structnames={"text_bbox_overlap_proxy":"文字bbox交叠","component_bbox_overlap_proxy":"元件bbox交叠","four_plus_arm_geometry_proxy":"≥4臂骨架几何","three_arm_geometry_proxy":"3臂骨架几何","ordinary_reference_tube":"普通GT参考邻域","remaining_background_unknown_semantics":"其余背景（语义未知）"}
    add(table(["互斥结构代理","Dice","P","R","FP / FN"],[[structnames[n],pct(r["dice"]),pct(r["precision"]),pct(r["recall"]) if r["tp"]+r["fn"] else "N/A",f"{r['fp']} / {r['fn']}"] for n,r in e["structure_proxies"].items()]))
    add("代理优先级：文字bbox > 元件bbox > ≥4臂 > 3臂 > GT2px参考邻域 > 其余背景。raw annotation bbox为原图顶左坐标；标准化component bbox另含底左坐标约定，不能直接混用。≥4臂区域Dice96.2519%、FN178；3臂FN1370，合计1548像素仅占全部FN4.9151%，现有代理不支持交叉点是主要漏检点。像素交叉不等于电气连接。")
    add("连接点dot、交叉是否相连、虚线/隐藏线、总线、图框/表格均无独立认证的全页语义标注，这些类别的全数据Dice/P/R **N/A**。没有用骨架分支冒充dot或用长线冒充总线。已有局部图明确展示dot（0287/0042）、箭头（0267/0079）、虚线图框（0208/0079）、表格（0170）；对应ROI数字见30张审查JSON，不能称全类得分。文字/元件bbox交叠也不等于被文字遮挡的真实wire。")
    p=e["reference_path_length"];c=e["components"]
    add(f"- 1548条≥10px、唯一物理草稿polyline总长{p['total_reference_length']:.2f}px；55条完全零覆盖，长{p['completely_missed_length']:.2f}px（{pct(p['completely_missed_length_fraction'])}）；覆盖≤10%占长度{pct(p['coverage_le10pct_length_fraction'])}，覆盖≤50%占{pct(p['coverage_le50pct_length_fraction'])}。这是中心线≤2px覆盖代理，不是全网召回。\n- 基线参考管道内连通1150/1548（74.2894%）。即使完全漏掉的长度不多，短断点也会破坏一整条路径；不能用Dice替代完整网络输出。路径真值仍是草稿，尚非电气金标准。\n- 预测8连通分量{c['all_prediction_components']}；距GT≤2px仍无接触的孤立分量{c['isolated_no_GT_within2_components']}、面积{c['isolated_no_GT_within2_area']}。面积≤4只有25块/66px，≤16为62块/433px，≤64为108块/1866px；主要FP并非小碎块。\n- 修正审计代码对“空预测”距离变换的特殊处理：0208的所有GT都应距预测无穷大；修正后总分区计数不变。源数据、模型无修改。")
    add("### 30张最差样本逐张局部审查\n\n验证只有26张，所以审查20张最差验证+10张最差训练，均查看原图/GT/误差局部对照；5张另查看整页。9/30发现明确局部路径/像素几何不符（验证4/20、训练5/10），6/30仍有wire/pin/符号范围歧义，2/30疑似宽度/小偏移；这些类别只记录各页的主要局部观察，会共存。最差样本有选择偏差，9/30不是全体标签错误率，更不是独立标注一致性或噪声上限。")
    add(f"逐页记录、坐标、局部TP/FP/FN：{link(OUT/'manual_review30.md')} · {link(OUT/'manual_review30.json')}。最差20页四联图完整清单：{link(OUT/'review30/manifest.json')}。\n\n![最差20页索引；详细请打开逐页文件]({str(OUT/'worst20_contact_sheet.png').replace(chr(92),'/')})")
    add(f"0170 P31.2347%、R92.5572%、Dice46.7073%，FP34798，占全部FP41.5513%、全部误差31.2985%。明确图框/logo ROI `[306,136,683,293]` 内GT为0、模型FP5713；这已经是可见的真非导线误检。只读leave-one-out分析Dice83.9541%，用于观察单页影响，**没有将0170从评测删除**。\n\n0208 ROI `[498,539,618,644]` 的GT547px全未检，GT骨架29.41%距任意灰度<220墨迹仍>2px，且倒U路线与原图不符；这页同时存在模型全空输出和标签问题，不能把0分都判为模型坏。\n\n详细例图：{link(OUT/'review30/0170/error_rois.png','0170图框误检')}、{link(OUT/'review30/0208/error_rois.png','0208路径不符')}、{link(OUT/'review30/0036/error_rois.png','0036真实漏线')}。")
    add("## 5. 根因排序与H1–H10证伪记录")
    add(table(["优先根因","数字证据","置信度","最小修复/验证","预期ΔDice","成本"],[
        ["非导线线形的语义区分不足","远FP66651；0170明确frame ROI有5713FP；多页IC边/绿色文字误检","高（存在）；总量语义归属仍不完整","先统一wire/pin/frame规则，再训练集确认难负样本","未知；去掉全部远FP的oracle为90.1726%，+10.7113点是不可直接达到的上界","标注核对约30–60min人工估算；配对采样实验约20min GPU估算"],
        ["草稿标签局部偏移、错误路线/缺段","最差30页9页明确几何不符，另6页语义待定；完整认证0","高（局部）；整体天花板未知","独立重标训练样本并保存第二份；不修改冻结val","未知，禁止把GT平移分数当提升预测","10页双审约30–60min人工估算，0 GPU"],
        ["泛化/训练后期波动","同尺度train90.3199% vs val79.4613%；best57→last60−2.3337点","中；未隔离数据量/BN/loss因素","固定epoch及单变量对照，模型选择移到训练内部校准","未知；不能保证GN/正则能追回10.86点","16epoch约9–11min/臂，据历史epoch时间估算"],
        ["局部方向/格点/上下文敏感","0036下移1px DOUT0/24→24/24；整批只+0.0022点、路径−10","高（局部存在）；作为整批主因证据不足","训练集固定方向/格点敏感性切片，之后只改池化或loss一项","本轮整页位移均值−0.0065点；架构修复未知","本轮全26页补实验36.89sec GPU/CPU总墙时"],
        ["阈值/后处理缺口","阈值最佳+0.4820；小块过滤最佳+0.0146；闭运算−0.0514","高（排除为主要瓶颈）","保持当前出口；训练内部校准后再独立验收","已测仅约0–0.5点且可能损失连通","本轮整套推理159.79sec总墙时"],
    ]))
    add("表中的unknown就是本轮不能估计；oracle只用GT计算理想误差消除上界，不是可部署方法或预期收益。新训练的数值门槛将作为预先声明的成功条件，不能改写成保证提升。")
    add(table(["假设","现有证据与结论","最小证伪/修复实验","ΔDice证据/预期","成本"],[
        ["H1 口径/阈值不匹配","micro79.4613、macro79.2210；最优阈值0.80为79.9432；不是主因","已完成同概率19阈值扫描；部署阈值需训练内校准","诊断+0.4820点，不采用","度量审计11.78sec，CPU"],
        ["H2 泄露","case先分split；跨split精确重复0、pHash≤8候选0；工程ID未知","已有case grouped Δ0；若取得工程ID，先审曝光再重新分组训练baseline","工程级Δ未知，不能靠换评测集估泄露程度","统计已完成；工程ID当前不可用"],
        ["H3 标签噪声/上限","9/30局部几何不符，独立双标注0；平移敏感不等于上限","训练页10张独立双审；保留两份并计算一致性，验证只读","上限/修标收益未知","30–60min人工估算，0 GPU"],
        ["H4 标签生成bug","131页wire只0/255；训练/默认推理无resize；17/26相关峰零；整体GT平移均更差","已排除软标签/默认mask插值/全局平移；剩下局部路径逐段核对","局部label修正收益未知","已有对齐与视觉检查，CPU"],
        ["H5 分辨率/下采样","默认已经原尺度滑窗；强制fit512降至46.7796%；4次maxpool存在但浅层线特征并非都消失","512/1024与均匀/高斯已完成；局部first-pool替换历史有因果证据，整批尚未隔离","1024均匀+0.3199、1024高斯+0.2267；无需声称恢复了默认被resize的线","各26页1024约7sec/配置"],
        ["H6 类不平衡/损失","train1.2662%、val1.3554%；现有loss已是BCE(pos8)+Dice+.5pin BCE(pos20)，不是纯BCE","先确认标签与负类；之后仅变负样本采样或仅加固定clDice权重，分别对照","新loss收益未知；未跑BCE/Tversky/Focal/Boundary训练，不伪造表","每臂16epoch约9–11min GPU估算"],
        ["H7 小batch/BN不稳","batch2、18个BN；last60整图77.1276，best57为79.4613；旋转模型换回v1 BN buffers从76.4987→70.4281","简单buffer回滚失败；不能据此排除权重/BN共适应。下一步只在train估BN或BN→GN单变量训练","回滚−6.0706点相对rotation；GN收益未知","last/BN回滚各26页约5sec；GN训练未执行"],
        ["H8 增广有害","原始与control无elastic/任意角/强几何缩放；唯一90°旋转臂比control低2.6060点","已做同步lossless90°配对；不再重复取消不存在的elastic","光度相对control+0.4119；旋转−2.6060","使用既有训练；本轮新训练0"],
        ["H9 过拟合/数据不足","同checkpoint整页train90.3199、val79.4613差10.8587；103/26页且style/工程ID不全","确认train标签和难负样本后做固定单变量对照；不加参数","数据量/正则的可追回幅度未知","分析完成；未来每臂约10min GPU估算"],
        ["H10 后处理不足","closing3−0.0514；remove≤4+0.0106；remove≤16+0.0146；线性度−0.2022；bridge−0.0009","固定参数、每项从baseline单独执行；已完成，没有叠加挑选","未支持“白拿1–4点”；桥接只多1参考路径","已完成，CPU，含在推理总墙时"],
    ]))
    add("BN在训练中更新running statistics、eval使用这些值；GN在train/eval均按当前输入分组计算统计，这是待验证的Norm候选依据，不能替代实验结果。[PyTorch BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)、[GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.GroupNorm.html)。梯度累积若仍每次前向batch2，不会自动使BN按累积后的大batch归一化（由前向统计机制推得）。clDice论文针对细长结构拓扑提供loss设计依据，但本项目没有它的训练结果；不能照搬论文收益或将栅格交叉自动当电气接点。[clDice原论文](https://arxiv.org/abs/2003.07311)。")
    add("## 6. 推理侧实验日志：每项单独对照")
    display_order=["native_uniform512","scale075","scale125","scale_mean","d4_mean","d4_multiscale_mean","gaussian512","uniform1024","gaussian1024","single_fit512","closing3","remove_area4","remove_area16","linear_components","endpoint_bridge2","ensemble_v1_control","ensemble_all4","v1_last","rotation_v1_BN_buffers","shift_x0_y1","shift_x1_y0","shift_x1_y1","phase_mean4"]
    inference_rows=[]
    for name in display_order:
        r=rows_by_name[name];best=r["diagnostic_best"]
        inference_rows.append([name,pct(r["micro"]["dice"]),pct(r["micro"]["precision"]),pct(r["micro"]["recall"]),f"{r['delta_dice_pp']:+.4f}",r["connected_paths"],f"{best['threshold']:.2f} / {pct(best['micro']['dice'])}" if best else "binary处理，不扫阈值"])
    add(table(["固定配置@0.5","micro Dice","P","R","Δ点","连通/1548","最佳阈值诊断（不采用）"],inference_rows))
    add("- 所有原图坐标上的评价均使用同一26页GT，源图/标签没有被修改。TTA概率在逆变换后等权均值，阈值固定0.5；不是对GT投票。0.75/1/1.25尺寸RGB bilinear、概率回原图bilinear；只有反事实fit512输入网格GT用nearest。\n- 默认512/stride384已经原尺度、约25%overlap；高斯sigma=size/8、最低权重0.001。1024/stride768改变patch大小，均匀1024→高斯1024才是纯融合权重比较（−0.0932点）。1024也改变上下文/训练尺寸分布，不能唯一归因为分辨率。\n- 后处理均从baseline开始，没有堆叠：3×3closing（padding保护图边）；8CC删除面积≤4/≤16；线性度保留骨架长≥4且bbox长宽比≥3或骨架长/√面积≥4；端点mutual nearest、距离≤3（中间缺≤2px）、双方切线≤30°、width1桥接。所有判断只用预测，无GT/元件真值指导。\n- 集成固定等权概率：v1+control和v1+control+contrast+rotation；没有按val调模型权重。四模型Dice+0.3601点但连通−90；两模型+0.1175点但连通−59。“集成有效”不能只看Dice。\n- 阈值扫描、所有变体均为固定审计实验，未依val挑生产方案。仍缺独立test，不报告泛化提升或官方分数。")
    add(table(["弱正收益的差值不确定性","ΔDice点","26页paired bootstrap95%区间"],[[n,f"{rows_by_name[n]['delta_dice_pp']:+.4f}",f"[{boot[n][0]:+.4f}, {boot[n][1]:+.4f}]"] for n in ("scale_mean","gaussian512","uniform1024","gaussian1024","ensemble_all4")]))
    add("bootstrap10000次、seed20261014，重采样单位是页；工程内相关未知未校正，因此这些是诊断区间，不能视为独立工程test置信区间。所有弱正收益区间都跨0。")
    add("### 0036局部机制与全批结果应分开\n\n本轮固定X1 ROI为y80、x620–680（61点）；DOUT为y181、x605–628（24点）。下移1px保留每个原像素、白色pad、逆裁回原坐标：DOUT p中位0.0174→0.7600，0/24→24/24；X1 p0.0388→0.0020，仍0/61。整批下移后的二值输出改变16060像素，Dice仅+0.0022点、路径1150→1140；4相位均值−0.0065点。输入相位敏感确实存在，但整体移动不是通用修复。该全页实验同时改变pool格点、tile起点与padding上下文，不能全部归因第一层pool。\n\n历史0036特征诊断在相同原像素的对齐局部浅层inc特征差0，替换深层up4分支复现DOUT变化、first-pool格点偏1也复现；所以“细线没有任何特征”并不成立。该结论局限于已诊断的ROI。历史X1旋转90°后61/61恢复，但同时增加U1非导线边误检且整条路径仍不连通；本轮D4均值X1只39/61、全批Dice下降，进一步排除了把单条响应升高直接当全网改善。")
    add(f"配置/逐页/阈值曲线：{link(OUT/'inference_protocol.json')} · {link(OUT/'inference_audit.json')}；相位补实验：{link(OUT/'phase_protocol.json')} · {link(OUT/'phase_audit.json')}。")
    add("## 7. 下一步三条行动：按投入产出排序\n\n这三条是有证伪条件的后续工作，不是本轮已经证明有效的训练配方。本轮只完成审计和推理实验。")
    followup=dict(seed=20261014,current_status="audit_complete_future_training_not_started",validation_policy="existing26 frozen read-only reporting only; no threshold/loss/checkpoint selection on them",independent_engineering_groups_available=False,
        label_review_train_cases=["0129","0088","0215","0254","0265","0073","0091","0079","0253","0281"],
        wire_policy_to_confirm=["physical wire vs component pin shaft","junction dot pixel extent and crossing semantics","arrow/net-label/flywire exclusions","frame/table/text negatives","actual path coordinates and nominal stroke width"],
        actions=[dict(rank=1,task="independent training-label review and semantic policy",GPU_minutes=0,expected_delta_Dice=None,success="record2 independent masks per sample and consistency by class; confirm draft path errors, no frozen val edits"),
            dict(rank=2,task="confirmed hard-negative crop sampling only",variable="training sampling distribution only; no loss/norm/augmentation change",design="matched control and treatment on same reviewed train labels; sampler25pct confirmed nonwire frame/body/text crops; all other draw/optimization settings fixed",epochs=16,expected_delta_Dice=None,GPU_minutes_estimate_per_arm=[9,11],success="train-internal calibration>=1pp micro gain with no relaxed-recall/path-connectivity decline; frozen26 descriptive reporting once",falsification="no reduction in confirmed negative-class FP or gain only from removing actual wire"),
            dict(rank=3,task="topology-specific loss ablation after label policy",variable="add0.1 soft clDice to unchanged wire loss only",epochs=16,expected_delta_Dice=None,GPU_minutes_estimate_per_arm=[9,11],success="train-internal calibration>=2pp reference connectivity gain and micro drop<=0.2pp; semantic crossings excluded from false merges",falsification="only larger tubes/false bridges or no gap/connectivity gain",architecture_followup="only if train-only phase slices remain dominant, separately compare first-pool alias handling; do not stack with loss")],
        training_rules="fresh matched controls from identical seeded RANDOM initialization; original v1 has seen all103 so do not warm-start then claim a reserved training subset is independent; same labels/batches/lr/epochs; fixed16 pilot and final checkpoint only; threshold calibration from train-internal split only; source/engineering grouping unresolved, never call it engineering grouped test; no new architecture capacity justification yet")
    write(OUT/"followup_protocol.json",followup)
    add(f"1. **先统一并核对训练标注的物理语义与位置。** 训练候选0129/0088/0215/0254/0265存在明确局部几何问题，0073有缺标，其余4页作为反例；双审保存独立mask及评审者信息。确认wire/pin杆、dot、箭头、图框负类规则，再统计一致性。不要修冻结val刷分，也不要自动把所有可见直线补为GT。准备表和复查入口：`scripts/write_unet_audit_report.py`生成 {link(OUT/'followup_protocol.json')}；局部证据从 {link(OUT/'manual_review30.md')} 打开。现有label生成位置：{link(ROOT/'src/pcb_parser/unet_data.py','unet_data.py',31)}；规则确认后才在**新训练标签目录**生成，不覆盖v1。\n\n2. **只验证“难负样本语义混淆”这一项。** 在确认是frame/body/text负类的训练ROI上增加固定25%采样份额，其他损失/BN/增强/初始权重/学习率/16轮完全相同，重新跑matched control与采样臂。采样代码位置：{link(ROOT/'src/pcb_parser/unet_retraining.py','unet_retraining.py',29)}；现有基线训练入口：{link(ROOT/'scripts/train_unet.py','train_unet.py',55)}。预期提升数值未知；预先成功条件是train内部校准micro至少+1点且relaxed recall与路径连通不降，失败条件为真负类FP没减少或靠误删实际导线得分。校准/选轮/选阈值都来自train内部、case重复组保持同侧；26页仅最终一次描述比较，工程组仍unknown。约9–11min/GPU/臂是历史时长估算。\n\n3. **针对真实断点，做单变量拓扑loss对照。** 标签规范明确后，在现有wire BCE+Dice上仅加入固定0.1 soft-clDice，不同时换GN/池化/增强；代码位置 {link(ROOT/'src/pcb_parser/unet_model.py','unet_model.py',29)}。预先成功条件：train内部参考连通率至少+2点且micro下降≤0.2点、无新增跨符号/交叉假连接；不满足就拒绝。收益未知。若训练侧已确认的格点切片仍是主要断点来源，另开仅改first-pool的独立对照，位置 {link(ROOT/'src/pcb_parser/vendor/pytorch_unet/unet_parts.py','unet_parts.py',34)}。暂不以0036一条线决定换大骨干。")
    add("训练动作2/3尚未实现或启动，所以这里提供真实代码改动位置和预注册方案，未写一个当前不存在的“直接运行新训练”命令。16轮是固定末轮的配对pilot，不按冻结val选轮；若从103训练页预留内部校准页，两臂须用相同seed随机初始化重新训练，因为原v1已经见过全部103页，不能warm-start后把这些预留页称为独立校准。内部划分使用新派生manifest，原划分文件冻结；完全相同图像组保持同侧。pilot有效后再预注册更长确认训练，不由26页结果决定候选权重或阈值。还没有工程ID/独立双标注，因此相应结论保持unknown。后续需要第二人标注时应以这份具体样本和规则为依据，不反复索取已确认不存在的文件。")
    add("## 8. 全部复现命令、验证与预算\n\n在PowerShell中按下面顺序执行。所有新增输出位于审计目录；预测阶段仅读取PNG与checkpoint，GT在预测结束后用于评价。脚本对冻结源文件校验SHA，默认模型不自动替换。")
    commands='''Set-Location -LiteralPath 'C:\\Users\\Pc\\Desktop\\vision\\pcb_parser_competition'
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_history.py
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_baseline.py --train-predictions
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_metrics.py
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_errors.py
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_manual_review.py
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_inference.py
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_phase.py
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\audit_unet_finalize.py
& .\\.venv-gpu\\Scripts\\python.exe -X utf8 scripts\\write_unet_audit_report.py'''
    add("```powershell\n"+commands+"\n```")
    add("已有概率缓存会加快复现。要完全重新推理，可将审计目录整体重命名备份后按顺序运行；不要删除训练源或模型。人工目视观察记录在脚本中是本次已查看图片后写下的固定审查结果，重新计算ROI统计不会产生第二人的独立标签。")
    add(table(["已执行阶段","记录墙时","硬件/目的"],[
        ["基线131页统计+103train/26val预测",f"{b['elapsed_seconds']:.2f}秒","RTX5060单卡/CPU"],
        ["度量/阈值/重复检查",f"{m['elapsed_seconds']:.2f}秒","CPU"],
        ["错误分区+30页出图（修正后重跑）",f"{e['elapsed_seconds']:.2f}秒","CPU"],
        ["22种基础推理/后处理/既有checkpoint对照",f"{a['elapsed_seconds']:.2f}秒","RTX5060单卡+CPU；0036 smoke另18.20秒"],
        ["4种相位补对照",f"{a['phase_addendum_elapsed_seconds']:.2f}秒","RTX5060单卡+CPU"],
        ["新训练",0,"本轮审计/推理投入占比100%；不含历史训练与人工阅读/编码时间"],
    ]))
    add(f"验证：baseline与原79.46%计数完全一致；control/contrast/rotation各自像素计数和路径数与历史独立评价完全一致；所有变体TP+FN=254422、26页/1548条参考路径不变；8种D4逆变换在合成网格上精确恢复；30张审查ID唯一；误差分区和等于115242；冻结{f['protected_files']}文件与默认checkpoint未改变。{link(OUT/'final_checks.json')}包含断言与bootstrap，{link(OUT/'artifact_manifest.json')}包含审计JSON/脚本SHA。")
    add("尚未完成且不可假装完成的量：工程级无泄露保证/分组test得分、独立双标注Dice/噪声天花板、全页认证的dot/虚线/总线/图框类分割得分，以及BCE/Tversky/Focal/clDice/Boundary/GN的单变量训练收益。上述每项均已给出缺口及最小后续验证设计。当前结论不需要新训练就成立。")
    (OUT/"report.md").write_text("\n".join(texts),encoding="utf-8")
    with (OUT/"inference_results.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["variant","pages","Dice_at0.5","precision","recall","delta_pp","connected_paths","total_paths","diagnostic_best_threshold","diagnostic_best_Dice","adopted"])
        for r in a["summary"]:
            best=r["diagnostic_best"];writer.writerow([r["variant"],r["pages"],r["micro"]["dice"],r["micro"]["precision"],r["micro"]["recall"],r["delta_dice_pp"],r["connected_paths"],r["total_paths"],best["threshold"] if best else "",best["micro"]["dice"] if best else "",False])
    print(str(OUT/"report.md"));assert_protected()


if __name__=="__main__":main()
