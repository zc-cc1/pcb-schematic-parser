/* EDA 精英挑战赛 决赛答辩 PPT builder (pptxgenjs, 16:9). */
const pptxgen = require("pptxgenjs");

const A = "E:/AI_Workplace/精英挑战赛/_audit_workspace/assets_build";
const D = "E:/AI_Workplace/精英挑战赛/_audit_workspace/deck_workspace";
const OUT = "E:/AI_Workplace/精英挑战赛/EDA_Final_Defense/EDA精英挑战赛_决赛答辩.pptx";

const W = 13.33, H = 7.5, M = 0.5;
const BG_D = "0B1F3A", PANEL = "EEF4FA", WHITE = "FFFFFF";
const PRIMARY = "123C6D", CYAN = "0FB5C9", CYAN_D = "0E7C8C", ORANGE = "F59E0B";
const INK = "17263B", MUTED = "5A6B80", LIGHT = "9FB3C8", CARD_D = "10305A";
const F = "Microsoft YaHei", MONO = "Consolas";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "EDA 精英挑战赛 参赛队";
pres.title = "AI 驱动的 PCB 原理图自动解析系统";

const shadow = () => ({ type: "outer", color: "1A2A44", blur: 7, offset: 2, angle: 45, opacity: 0.18 });
const soft = () => ({ type: "outer", color: "8899AA", blur: 5, offset: 1, angle: 90, opacity: 0.14 });

function fit(pw, ph, bx, by, bw, bh) { // contain-fit, centered
  const r = Math.min(bw / pw, bh / ph);
  const w = pw * r, h = ph * r;
  return { x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h };
}
function img(slide, path, pw, ph, bx, by, bw, bh, opts = {}) {
  const g = fit(pw, ph, bx, by, bw, bh);
  slide.addImage({ path, x: g.x, y: g.y, w: g.w, h: g.h, shadow: opts.shadow ? shadow() : undefined });
  return g;
}
function header(slide, title, sub = "") {
  slide.addText(title, { x: M, y: 0.28, w: 10.6, h: 0.62, fontSize: 27, bold: true, color: PRIMARY, fontFace: F, margin: 0 });
  if (sub) slide.addText(sub, { x: M, y: 0.88, w: 12.3, h: 0.32, fontSize: 12.5, color: MUTED, fontFace: F, margin: 0 });
}
function pageNo(slide, n, dark = false) {
  slide.addText(String(n), { x: W - 0.75, y: H - 0.42, w: 0.5, h: 0.3, fontSize: 10, color: dark ? LIGHT : MUTED, align: "right", fontFace: F, margin: 0 });
}
function footer(slide, text) {
  slide.addText(text, { x: M, y: H - 0.44, w: 11.6, h: 0.3, fontSize: 9.5, color: MUTED, fontFace: F, margin: 0 });
}
function card(slide, x, y, w, h, fill = WHITE, line = "DCE5EF") {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line, width: 1 }, rectRadius: 0.06, shadow: soft() });
}
function chip(slide, x, y, w, h, big, small, bigColor = ORANGE, dark = false, bigSize = 20) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: dark ? CARD_D : WHITE }, line: { color: dark ? "1E4D8F" : "DCE5EF", width: 1 }, rectRadius: 0.07, shadow: soft() });
  slide.addText(big, { x: x + 0.08, y: y + 0.06, w: w - 0.16, h: h * 0.5, fontSize: bigSize, bold: true, color: bigColor, fontFace: F, align: "center", valign: "middle", margin: 0 });
  slide.addText(small, { x: x + 0.08, y: y + h * 0.52, w: w - 0.16, h: h * 0.42, fontSize: 10, color: dark ? LIGHT : MUTED, fontFace: F, align: "center", valign: "top", margin: 0 });
}
function chevronFlow(slide, x, y, w, h, items, gap = 0.14) {
  const n = items.length;
  const cw = (w - gap * (n - 1)) / n;
  items.forEach((it, i) => {
    const cx = x + i * (cw + gap);
    slide.addShape(pres.shapes.CHEVRON, { x: cx, y, w: cw, h, fill: { color: it[2] || PANEL }, line: { color: "C9D6E6", width: 0.75 } });
    slide.addText(it[0], { x: cx + 0.12, y: y + 0.05, w: cw - 0.28, h: h * 0.48, fontSize: it[3] || 10.5, bold: true, color: INK, fontFace: F, align: "center", valign: "middle", margin: 0 });
    slide.addText(it[1], { x: cx + 0.16, y: y + h * 0.5, w: cw - 0.3, h: h * 0.44, fontSize: 8.2, color: MUTED, fontFace: F, align: "center", valign: "top", margin: 0 });
  });
}

/* ============ Slide 1 · 封面 ============ */
{
  const s = pres.addSlide();
  s.background = { color: BG_D };
  // right PCB art panel (868x1155, ratio 0.7515 -> full height 7.5 => w 5.636)
  const gw = 7.5 * (868 / 1155);
  s.addImage({ path: `${D}/d_cover_pcb.png`, x: W - gw, y: 0, w: gw, h: 7.5 });
  s.addText("中国研究生创“芯”大赛 · EDA 精英挑战赛 · 赛题六 决赛答辩", { x: 0.62, y: 0.62, w: 6.6, h: 0.35, fontSize: 12.5, color: CYAN, fontFace: F, margin: 0 });
  s.addText("AI 驱动的 PCB 原理图\n自动解析系统", { x: 0.62, y: 1.28, w: 6.9, h: 2.2, fontSize: 40, bold: true, color: WHITE, fontFace: F, margin: 0, lineSpacingMultiple: 1.08 });
  s.addText("从原理图图像到结构化电气网络的自动化解析", { x: 0.62, y: 3.62, w: 6.7, h: 0.4, fontSize: 15.5, color: LIGHT, fontFace: F, margin: 0 });
  // score chips
  chip(s, 0.62, 4.42, 2.02, 1.24, "74 分", "官方评测成绩", ORANGE, true, 26);
  chip(s, 2.82, 4.42, 2.02, 1.24, "第 47 名", "全国排名", WHITE, true, 20);
  chip(s, 5.02, 4.42, 2.02, 1.24, "前 61 强", "成功晋级决赛", WHITE, true, 20);
  s.addShape(pres.shapes.LINE, { x: 0.62, y: 6.18, w: 5.9, h: 0, line: { color: "22456F", width: 1 } });
  s.addText([
    { text: "全本地运行 · 零 LLM Token", options: { color: CYAN, bold: true } },
    { text: "     参赛队:________      2026 年 11 月", options: { color: LIGHT } },
  ], { x: 0.62, y: 6.36, w: 6.9, h: 0.4, fontSize: 12, fontFace: F, margin: 0 });
  s.addText("成绩与排名以官方平台记录为准(答辩前补充成绩页截图)", { x: 0.62, y: 6.82, w: 6.9, h: 0.3, fontSize: 9.5, color: "5E7A9C", fontFace: F, margin: 0 });
}

/* ============ Slide 2 · 赛题分析与技术挑战 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "赛题分析与技术挑战", "输入一张原理图 PNG,输出符合官方 JSON 契约的结构化结果");
  // task I/O strip
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.32, w: 7.3, h: 0.86, fill: { color: PANEL }, line: { color: "D5E2F0", width: 1 }, rectRadius: 0.06 });
  s.addText("PNG 输入", { x: 0.78, y: 1.46, w: 1.5, h: 0.32, fontSize: 13, bold: true, color: PRIMARY, fontFace: F, margin: 0 });
  s.addText("官方数据集 PCbnet · 30~50 案例", { x: 0.78, y: 1.78, w: 2.4, h: 0.3, fontSize: 9, color: MUTED, fontFace: F, margin: 0 });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 3.3, y: 1.6, w: 0.85, h: 0.34, fill: { color: CYAN } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 4.32, y: 1.44, w: 3.3, h: 0.64, fill: { color: WHITE }, line: { color: PRIMARY, width: 1.2 }, rectRadius: 0.06 });
  s.addText([
    { text: "components", options: { color: CYAN_D, bold: true } }, { text: "  ", options: {} },
    { text: "pins", options: { color: PRIMARY, bold: true } }, { text: "  ", options: {} },
    { text: "nets", options: { color: ORANGE, bold: true } },
  ], { x: 4.5, y: 1.5, w: 3.0, h: 0.5, fontSize: 14, fontFace: MONO, valign: "middle", margin: 0 });
  // four difficulties (2x2)
  const diffs = [
    ["1  密集元器件检测", "高密度排布 · 小型贴片器件易漏检/类别混淆"],
    ["2  引脚定位与元件归属", "多引脚 IC 引脚错位、漏识别、绑定错误"],
    ["3  文字识别与型号关联", "文字扎堆、倾斜标注 → 位号/参数错配"],
    ["4  复杂拓扑与连通性", "T 型/十字节点、多分支、断线 → 错连/漏连/虚假网络"],
  ];
  diffs.forEach((d, i) => {
    const x = M + (i % 2) * 3.75, y = 2.5 + Math.floor(i / 2) * 1.62;
    card(s, x, y, 3.55, 1.42);
    s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: y + 0.2, w: 0.16, h: 0.16, fill: { color: i === 3 ? ORANGE : CYAN } });
    s.addText(d[0], { x: x + 0.44, y: y + 0.12, w: 3.0, h: 0.34, fontSize: 12.5, bold: true, color: INK, fontFace: F, margin: 0 });
    s.addText(d[1], { x: x + 0.44, y: y + 0.5, w: 3.0, h: 0.8, fontSize: 10, color: MUTED, fontFace: F, margin: 0 });
  });
  s.addText("官方列出的四大技术难点,即本方案的全部设计输入", { x: M, y: 5.88, w: 7.3, h: 0.34, fontSize: 11.5, bold: true, color: PRIMARY, fontFace: F, margin: 0 });
  // right: weights chart + takeaway
  card(s, 8.15, 1.32, 4.68, 4.05, WHITE);
  const g1 = fit(1180, 600, 8.3, 1.44, 4.38, 2.9);
  s.addImage({ path: `${D}/d_score_weights.png`, x: g1.x, y: g1.y, w: g1.w, h: g1.h });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.34, y: 4.42, w: 4.3, h: 0.82, fill: { color: "FFF7E9" }, line: { color: ORANGE, width: 1 }, rectRadius: 0.06 });
  s.addText("技术路线围绕官方评分标准设计:权重 0.35 的网络超图是主战场;5px / 20px 硬阈值决定像素级精度要求", { x: 8.5, y: 4.5, w: 4.0, h: 0.68, fontSize: 9.5, color: INK, fontFace: F, margin: 0 });
  s.addText("FinalScore = FinalScore_Accuracy(100) + FinalScore_Data(10)\nAccuracy = 0.95 × 案例精度 + 0.05 × (m/M) × Token 赋分", { x: 8.15, y: 5.52, w: 4.68, h: 0.62, fontSize: 9.5, color: MUTED, fontFace: F, margin: 0 });
  footer(s, "口径:赛题指南(30 页)评分细则与四大技术难点;数据集 PCbnet(ECNUACRush)");
  pageNo(s, 2);
}

/* ============ Slide 3 · 总体架构 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "系统总体架构", "两级编排:S 阶段子进程链(S14→S24)+ agent.py 证据门控修复链");
  img(s, `${A}/arch_pipeline.png`, 1831, 1258, 0.6, 1.3, 12.15, 5.35, { shadow: true });
  footer(s, "每阶段标注模型/算法、输入输出、代码入口;零 LLM Token · 全离线 · 逐阶段哈希锁定与审计日志 · 失败回退最近有效预测");
  pageNo(s, 3);
}

/* ============ Slide 4 · 视觉感知 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "视觉感知:多模型协同,各管一件事", "检测器找元件(框 CAD 单元),U-Net 找导线与引脚尖(像素级),OCR 读文字 —— 输出均为候选,结论留给证据体系");
  // left: yolo fusion diagram + F1 chip
  const g1 = fit(1187, 717, 0.42, 1.34, 6.6, 3.6);
  s.addImage({ path: `${D}/d_yolo_fusion.png`, x: g1.x, y: g1.y, w: g1.w, h: g1.h });
  chip(s, 0.62, 5.14, 2.5, 1.06, "F1 0.883", "检测 59 例微平均(fused 模式胜出)", ORANGE, false, 19);
  chip(s, 3.32, 5.14, 3.5, 1.06, "35 类自训检测器", "imgsz1536 · 120 epochs · 由 v2 热启动", PRIMARY, false, 14);
  // right: real U-Net operating overlay
  const g2 = fit(1190, 915, 7.28, 1.28, 5.62, 3.9);
  card(s, 7.2, 1.2, 5.75, 4.0, WHITE);
  s.addImage({ path: `${D}/d_unet_wire_panel.png`, x: g2.x, y: g2.y, w: g2.w, h: g2.h });
  s.addText("真实管线输出:输入图 + wire 概率图(阈值 0.35;pin 头另解码出 431 个候选)", { x: 7.2, y: 5.24, w: 5.75, h: 0.3, fontSize: 9.5, color: MUTED, fontFace: F, align: "center", margin: 0 });
  // model badges
  const badges = [
    ["YOLO nano", "自研训练", CYAN_D], ["WirePinUNet 双头", "自研训练(milesial 积木)", PRIMARY], ["PP-OCRv6 OCR", "直接调用 · 离线 ONNX", MUTED],
  ];
  badges.forEach((b, i) => {
    const x = 0.62 + i * 2.28;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 6.36, w: 2.16, h: 0.62, fill: { color: PANEL }, line: { color: "C9D6E6", width: 0.75 }, rectRadius: 0.08 });
    s.addText(b[0], { x: x + 0.08, y: 6.4, w: 2.0, h: 0.3, fontSize: 10, bold: true, color: INK, fontFace: F, margin: 0 });
    s.addText(b[1], { x: x + 0.08, y: 6.68, w: 2.0, h: 0.26, fontSize: 8.2, color: b[2], fontFace: F, margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.2, y: 6.36, w: 5.75, h: 0.62, fill: { color: "F6F8FB" }, line: { color: "DCE5EF", width: 0.75 }, rectRadius: 0.08 });
  s.addText("JunctionCNN(自训)已完成训练,作为可选增强保留;默认管线采用保守结点策略(判错代价 > 漏连)", { x: 7.34, y: 6.42, w: 5.5, h: 0.52, fontSize: 8.8, color: INK, fontFace: F, margin: 0 });
  footer(s, "口径:检测 F1 来自 configs/baseline_v3_adopted.json(59 例微平均);分割图为 keep-work 运行真实产物;权重 sha256 全部锁定校验");
  pageNo(s, 4);
}

/* ============ Slide 5 · 拓扑重建 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "导线拓扑与电气网络重建", "U-Net 概率图 → 骨架 → 线段 → 物理网络;结点处“宁漏连、不错连”");
  const gl = fit(1190, 845, 0.45, 1.3, 6.1, 3.7);
  card(s, 0.4, 1.24, 6.2, 3.82, WHITE);
  s.addImage({ path: `${A}/fig3c_topology_real.png`, x: gl.x, y: gl.y, w: gl.w, h: gl.h });
  s.addText("骨架 7,210 px + 39 个结点(度≥3 聚类,洋红)", { x: 0.4, y: 5.1, w: 6.2, h: 0.3, fontSize: 9.5, color: MUTED, fontFace: F, align: "center", margin: 0 });
  const gr = fit(1190, 845, 6.75, 1.3, 6.1, 3.7);
  card(s, 6.7, 1.24, 6.2, 3.82, WHITE);
  s.addImage({ path: `${A}/fig3d_graph_networks_real.png`, x: gr.x, y: gr.y, w: gr.w, h: gr.h });
  s.addText("管线物理网络候选:238 网 / 362 折线边(按网着色,合并前中间表示)", { x: 6.7, y: 5.1, w: 6.2, h: 0.3, fontSize: 9.5, color: MUTED, fontFace: F, align: "center", margin: 0 });
  // four handling chips
  const cases = [
    ["T 型 / 十字", "保守拆分,显式不确定;学习型分类器留作增强"],
    ["断线", "仅凭硬证据恢复:60px 共线重合 或 红点+95% 墨迹"],
    ["飞线", "同名非电源标签逻辑合并,edges 为零长度段"],
    ["VCC / GND", "四道门才合并电源网;不画物理长线"],
  ];
  cases.forEach((c, i) => {
    const x = 0.45 + i * 2.74;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.62, w: 2.62, h: 1.02, fill: { color: PANEL }, line: { color: "C9D6E6", width: 0.75 }, rectRadius: 0.07 });
    s.addText(c[0], { x: x + 0.12, y: 5.7, w: 2.38, h: 0.3, fontSize: 10.5, bold: true, color: PRIMARY, fontFace: F, margin: 0 });
    s.addText(c[1], { x: x + 0.12, y: 6.0, w: 2.38, h: 0.6, fontSize: 8.2, color: INK, fontFace: F, margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 11.55, y: 5.62, w: 1.28, h: 1.02, fill: { color: PRIMARY }, line: { color: PRIMARY, width: 1 }, rectRadius: 0.07 });
  s.addText("238 → 63", { x: 11.55, y: 5.68, w: 1.28, h: 0.5, fontSize: 14, bold: true, color: WHITE, fontFace: F, align: "center", margin: 0 });
  s.addText("物理候选→逻辑网", { x: 11.55, y: 6.16, w: 1.28, h: 0.44, fontSize: 7.4, color: LIGHT, fontFace: F, align: "center", margin: 0 });
  footer(s, "口径:238 物理网络候选为建图阶段中间表示(wire_graph),63 为最终输出逻辑网络(含飞线/电源合并),勿混用;图为官方示例页真实运行产物");
  pageNo(s, 5);
}

/* ============ Slide 6 · 核心亮点 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "核心亮点:证据门控 × 事务式修复", "让每一处“改写”都有证据、可回滚 —— 110+ 修复模块的协同纪律");
  // left: page13 case
  const gl = fit(1700, 980, 0.42, 1.26, 7.35, 3.62);
  card(s, 0.36, 1.2, 7.45, 3.76, WHITE);
  s.addImage({ path: `${A}/fig4_page13_repair_case.png`, x: gl.x, y: gl.y, w: gl.w, h: gl.h });
  // numbers row under fig4
  s.addText([
    { text: "页13 导线修复  ", options: { bold: true, color: INK } },
    { text: "32/28/17 → 46/13/3   ", options: { color: PRIMARY, bold: true } },
    { text: "F1 58.72% → ", options: { color: INK } },
    { text: "85.19%", options: { color: ORANGE, bold: true } },
    { text: "   总分 82.871 → 92.656", options: { color: INK } },
  ], { x: 0.42, y: 5.02, w: 7.4, h: 0.4, fontSize: 12.5, fontFace: F, margin: 0 });
  // right: SW1/SW2 case
  const gr = fit(560, 422, 8.0, 1.26, 5.0, 3.2);
  card(s, 7.95, 1.2, 5.1, 3.76, WHITE);
  s.addImage({ path: `${A}/fig5_evidence_gated_ocr.png`, x: gr.x, y: gr.y, w: gr.w, h: gr.h });
  s.addText("案例二 型号恢复(v102 实测)", { x: 8.1, y: 4.56, w: 4.8, h: 0.3, fontSize: 11, bold: true, color: INK, fontFace: F, margin: 0 });
  s.addText([
    { text: "SW1/SW2:Name 由错误回退值 ", options: { color: MUTED } },
    { text: "“SW1”→“PTS8153JK250SMTRLFS”", options: { color: PRIMARY, bold: true } },
    { text: "(邻近文本堆栈 · 旋转视角 · 置信 0.998/0.992)", options: { color: MUTED } },
  ], { x: 8.1, y: 4.86, w: 4.85, h: 0.62, fontSize: 9.5, fontFace: F, margin: 0 });
  // bottom mechanism chevrons
  chevronFlow(s, 0.42, 5.62, 12.45, 1.14, [
    ["多视角 OCR", "2/3/4× × 0/90°,预算 12~160 次/页"],
    ["置信度门控", "≥2 视角一致且置信≥0.98"],
    ["几何/拓扑约束", "成员只增不减 · NC 禁入网"],
    ["冲突检测", "截断字形/拼写冲突一律拒收"],
    ["原子事务提交", "提案冲突全方否决,预算 64/16"],
    ["失败回滚", "remember() + 仅读本图恢复"],
  ]);
  footer(s, "口径:案例一数字出自 docs/network_junction_reconciliation.md(修复前快照 page13_status_20261020/status.json);案例二出自本机 v102 冒烟 export_semantics 审计");
  pageNo(s, 6);
}

/* ============ Slide 7 · 输出校准 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "输出校准:面向官方标注规则", "统计校准而非查表回填 —— 200 页 9186 组件测出官方画法约定");
  // left three cards
  const rows = [
    ["① 类型别名与字段规范化", "resistor/capacitor/… → r/c/l/d/box/led/v(提交 8180 · 145 元件验证)\nName==key 公开约定:r 1.000 / c .999 / box 仅 .242 → box 保留型号", CYAN_D],
    ["② bbox 几何校准(两套口径,分别成立)", "canonical 框约定(40 页公开子集):达标率 0.905 → 1.000\n按页缩放(13 页 released):20px 内 150/163 → 159/163", PRIMARY],
    ["③ 引脚尖端坐标修正", "尖端 18px 吸附(实测 16~22px 平台区)+ 1000 型号引脚库映射", ORANGE],
  ];
  rows.forEach((r, i) => {
    const y = 1.3 + i * 1.62;
    card(s, 0.45, y, 7.5, 1.44);
    s.addText(r[0], { x: 0.62, y: y + 0.08, w: 7.2, h: 0.34, fontSize: 12.5, bold: true, color: INK, fontFace: F, margin: 0 });
    s.addText(r[1], { x: 0.62, y: y + 0.46, w: 7.2, h: 0.9, fontSize: 10, color: MUTED, fontFace: F, margin: 0 });
    s.addShape(pres.shapes.OVAL, { x: 7.55, y: y + 0.62, w: 0.18, h: 0.18, fill: { color: r[2] } });
  });
  // right: json structure + U3 callout
  const g1 = fit(1133, 733, 8.1, 1.28, 4.9, 3.2);
  s.addImage({ path: `${D}/d_json_structure.png`, x: g1.x, y: g1.y, w: g1.w, h: g1.h });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.15, y: 4.72, w: 4.85, h: 1.9, fill: { color: "FFF7E9" }, line: { color: ORANGE, width: 1.2 }, rectRadius: 0.07, shadow: soft() });
  s.addText("1~2 px", { x: 8.15, y: 4.86, w: 4.85, h: 0.72, fontSize: 34, bold: true, color: ORANGE, align: "center", fontFace: F, margin: 0 });
  s.addText("U3(RP2350A)bbox 与官方示例标注之差\n本系统 [539,281,689,731] vs 官方 [540,280,690,730]\n(官方 20px 容差 · 官方示例页实测)", { x: 8.3, y: 5.62, w: 4.55, h: 0.9, fontSize: 9.5, color: INK, align: "center", fontFace: F, margin: 0 });
  footer(s, "口径:canonical 与页缩放是两套独立实验(不同数据集、不同机制),均生效但不可相乘;库映射 PinF1 0.031→0.549 见 defense_supplement §三");
  pageNo(s, 7);
}

/* ============ Slide 8 · 实验验证 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "多层次实验验证", "官方评测 · 公开集回放 · 本机端到端复现 —— 三层证据,口径分明");
  const g1 = fit(1095, 594, 0.42, 1.3, 7.3, 4.0);
  s.addImage({ path: `${D}/d_three_tiers.png`, x: g1.x, y: g1.y, w: g1.w, h: g1.h });
  // key metrics list
  card(s, 7.95, 1.32, 4.9, 3.95, WHITE);
  s.addText("关键指标", { x: 8.15, y: 1.44, w: 4.5, h: 0.34, fontSize: 13, bold: true, color: PRIMARY, fontFace: F, margin: 0 });
  const mets = [
    ["检测 F1(59 例微平均)", "0.883", ""],
    ["PinF1 · 引脚库映射", "0.031 → 0.549", ORANGE],
    ["页13 导线 F1(结点归并)", "58.72% → 85.19%", ORANGE],
    ["bbox 达标率(canonical·40 页)", "0.905 → 1.000", PRIMARY],
    ["页5 连通率(端子拓扑)", "84.96% → 100%", PRIMARY],
  ];
  mets.forEach((m, i) => {
    const y = 1.86 + i * 0.66;
    s.addShape(pres.shapes.LINE, { x: 8.15, y: y + 0.56, w: 4.5, h: 0, line: { color: "E4EAF2", width: 0.75 } });
    s.addText(m[0], { x: 8.15, y, w: 3.0, h: 0.5, fontSize: 10, color: INK, fontFace: F, valign: "middle", margin: 0 });
    s.addText(m[1], { x: 10.7, y, w: 2.0, h: 0.5, fontSize: 12.5, bold: true, color: m[2] || PRIMARY, fontFace: F, valign: "middle", align: "right", margin: 0 });
  });
  // U3 comparison strip
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.66, w: 12.35, h: 1.0, fill: { color: PANEL }, line: { color: "C9D6E6", width: 1 }, rectRadius: 0.07 });
  s.addText([
    { text: "逐例对照(官方示例页)   ", options: { bold: true, color: INK, fontSize: 12 } },
    { text: "官方标注 U3 bbox [540,280,690,730]   vs   本系统 [539,281,689,731]   →   ", options: { color: INK } },
    { text: "误差 1~2 px", options: { color: ORANGE, bold: true, fontSize: 14 } },
    { text: "(20px 容差)   冒烟输出 89 components / 165 pins / 63 nets", options: { color: MUTED } },
  ], { x: 0.72, y: 5.78, w: 11.9, h: 0.76, fontSize: 11, fontFace: F, valign: "middle", margin: 0 });
  footer(s, "口径:98.93 为 13 页公开集开发评估分(v101-score),不与官方 74 分直接比较;冒烟为本机 CPU 端到端复现,输出与决策日志已归档");
  pageNo(s, 8);
}

/* ============ Slide 9 · 工程实现 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "工程实现与系统可靠性", "从算法实验到可交付的工程系统:可追溯、可复现、可降级");
  // left feature rows
  const feats = [
    ["Docker 容器化交付", "官方基础镜像 + CUDA 12.1 · 入口 run.sh 已写入镜像 config", CYAN_D],
    ["全离线 · 零 LLM Token", "未调用赛会 qwen3.7-plus;Token 赋分取最优区间", PRIMARY],
    ["SHA256 全链校验", "249 项逐文件 manifest · 3 权重指纹锁定 · 预检 12 项资产", PRIMARY],
    ["分阶段审计日志", "每图 20+ 份 JSON(前后 bbox/证据/耗时),逐阶段可回放", CYAN_D],
    ["输出合法性校验", "validate_target + sanitize 只删无效对象,绝不推断内容", PRIMARY],
    ["626+ 单元测试", "42 文件:哈希/预算/幂等/不可变契约;构建期 overlay_check 门禁", ORANGE],
  ];
  feats.forEach((f, i) => {
    const y = 1.32 + i * 0.86;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.06, w: 0.3, h: 0.3, fill: { color: f[2] } });
    s.addText(String(i + 1), { x: 0.5, y: y + 0.06, w: 0.3, h: 0.3, fontSize: 11, bold: true, color: WHITE, align: "center", fontFace: F, margin: 0 });
    s.addText(f[0], { x: 0.98, y, w: 5.9, h: 0.36, fontSize: 13, bold: true, color: INK, fontFace: F, margin: 0 });
    s.addText(f[1], { x: 0.98, y: y + 0.36, w: 6.2, h: 0.4, fontSize: 9.5, color: MUTED, fontFace: F, margin: 0 });
    if (i < 5) s.addShape(pres.shapes.LINE, { x: 0.98, y: y + 0.78, w: 6.2, h: 0, line: { color: "EDF1F7", width: 0.75 } });
  });
  // right: recovery flow + output tree + honest note
  const g1 = fit(1156, 563, 7.3, 1.3, 5.65, 2.6);
  s.addImage({ path: `${D}/d_recovery_flow.png`, x: g1.x, y: g1.y, w: g1.w, h: g1.h });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.35, y: 4.1, w: 5.6, h: 1.42, fill: { color: "0D2237" }, line: { color: "1E4D8F", width: 1 }, rectRadius: 0.07 });
  s.addText([
    { text: "/output\n", options: { color: CYAN, bold: true } },
    { text: "├ decision_log.json          逐例状态/耗时/错误\n", options: { color: "C9D9EA" } },
    { text: "├ final_results/<case>_Pred.json   最终提交\n", options: { color: "C9D9EA" } },
    { text: "└ logs/<case>_*.json        20+ 阶段审计", options: { color: "C9D9EA" } },
  ], { x: 7.55, y: 4.22, w: 5.25, h: 1.2, fontSize: 9.5, fontFace: MONO, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.35, y: 5.7, w: 5.6, h: 1.0, fill: { color: "FFF7E9" }, line: { color: ORANGE, width: 1 }, rectRadius: 0.07 });
  s.addText("运行验证:CPU 端到端 2000s/例(官方示例页);官方 2h 时限按 GPU 3 并发设计,官方机时未实测 —— 不据此宣称达标", { x: 7.52, y: 5.8, w: 5.3, h: 0.8, fontSize: 9.5, color: INK, fontFace: F, margin: 0 });
  footer(s, "第 6 次评测曾因入口打包得 0 分 → 此后每次提交附构建期断言(overlay_check.py)与逐文件哈希清单(package_manifest.json)");
  pageNo(s, 9);
}

/* ============ Slide 10 · 成绩与总结(dark) ============ */
{
  const s = pres.addSlide();
  s.background = { color: BG_D };
  s.addText("项目成果与未来展望", { x: M, y: 0.32, w: 9, h: 0.6, fontSize: 27, bold: true, color: WHITE, fontFace: F, margin: 0 });
  // left scores
  chip(s, 0.6, 1.3, 2.1, 1.5, "74 分", "官方评测成绩", ORANGE, true, 30);
  chip(s, 2.88, 1.3, 2.1, 1.5, "第 47 名", "全国排名", WHITE, true, 21);
  chip(s, 5.16, 1.3, 2.1, 1.5, "前 61 强", "晋级决赛", WHITE, true, 21);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 3.0, w: 6.66, h: 1.06, fill: { color: CARD_D }, line: { color: "1E4D8F", width: 1 }, rectRadius: 0.07 });
  s.addText([
    { text: "版本迭代:  ", options: { color: LIGHT } },
    { text: "28.92 分", options: { color: WHITE, bold: true } },
    { text: "(旧版 8422)  ", options: { color: LIGHT, fontSize: 10 } },
    { text: "→  74 分", options: { color: ORANGE, bold: true, fontSize: 18 } },
    { text: "  最终提交(+45)", options: { color: LIGHT, fontSize: 10 } },
  ], { x: 0.85, y: 3.16, w: 6.2, h: 0.74, fontSize: 15, fontFace: F, valign: "middle", margin: 0 });
  s.addText("零 LLM Token(token_usage=0)· 全本地推理 · 3 个自训视觉模型 + 110+ 证据门控修复模块", { x: 0.62, y: 4.3, w: 6.7, h: 0.4, fontSize: 11, color: LIGHT, fontFace: F, margin: 0 });
  s.addText("诚实边界:官方分含 2/3 黑盒难例,泛化是当前主要差距 —— 也是下一步投入方向", { x: 0.62, y: 4.72, w: 6.7, h: 0.4, fontSize: 10.5, color: "6E86A6", fontFace: F, margin: 0 });
  // right contributions + outlook
  s.addText("三项核心贡献", { x: 7.75, y: 1.28, w: 5, h: 0.4, fontSize: 14, bold: true, color: CYAN, fontFace: F, margin: 0 });
  const contribs = [
    "自训视觉感知:YOLO nano / 双头 U-Net / 结点 CNN(全部 sha256 可溯)",
    "证据门控的保守修复体系:0.98×2 视角门控 + 事务式引擎 + 失败回滚",
    "面向评分规则的结构化输出校准:bbox 约定 / Name==key / 引脚库映射",
  ];
  contribs.forEach((c, i) => {
    const y = 1.78 + i * 0.78;
    s.addShape(pres.shapes.OVAL, { x: 7.78, y: y + 0.02, w: 0.26, h: 0.26, fill: { color: ORANGE } });
    s.addText(String(i + 1), { x: 7.78, y: y + 0.02, w: 0.26, h: 0.26, fontSize: 10, bold: true, color: WHITE, align: "center", fontFace: F, margin: 0 });
    s.addText(c, { x: 8.16, y: y - 0.04, w: 4.8, h: 0.72, fontSize: 10.5, color: "DCE6F2", fontFace: F, margin: 0 });
  });
  s.addText("未来展望", { x: 7.75, y: 4.28, w: 5, h: 0.4, fontSize: 14, bold: true, color: CYAN, fontFace: F, margin: 0 });
  const outs = ["提升黑盒难例泛化(结点分类器转正 · 域适配)", "扩充复杂原理图数据与难例标注", "优化复杂拓扑与密集引脚识别"];
  outs.forEach((c, i) => {
    const y = 4.74 + i * 0.62;
    s.addShape(pres.shapes.OVAL, { x: 7.82, y: y + 0.1, w: 0.12, h: 0.12, fill: { color: "3E7FB8" } });
    s.addText(c, { x: 8.06, y, w: 4.9, h: 0.5, fontSize: 10.5, color: "DCE6F2", fontFace: F, margin: 0 });
  });
  s.addShape(pres.shapes.LINE, { x: 0.6, y: 5.62, w: 6.66, h: 0, line: { color: "22456F", width: 1 } });
  s.addText("谢谢各位评委 —— 完整证据链见答辩材料(docs/eda_final_defense/)", { x: 0.6, y: 5.8, w: 8, h: 0.4, fontSize: 12, color: LIGHT, fontFace: F, margin: 0 });
  s.addText("PCB 原理图 → 结构化电气网络:让每一条导线都有证据", { x: 0.6, y: 6.3, w: 9, h: 0.5, fontSize: 16, bold: true, color: WHITE, fontFace: F, margin: 0 });
  pageNo(s, 10, true);
}

/* ============ B1 · 架构与模块说明 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "备份 A · 核心模块关系", "模型产生候选与概率 → 证据体系“担保” → 规则引擎保守修复 → 适配器对齐官方约定 → 可审计导出");
  img(s, `${A}/arch_modules.png`, 1831, 1189, 0.6, 1.28, 12.15, 5.4, { shadow: true });
  footer(s, "视觉模型层 / OCR 证据体系 / 几何拓扑处理 / 规则修复引擎 / 字段配对修复 / 输出适配器 —— 全部模块与 15 个 finalize 阶段的对应关系见 source_audit.md §5");
  pageNo(s, "B1");
}

/* ============ B2 · 训练与数据来源 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "备份 B · 模型训练与数据来源", "全部推理权重随包携带并 sha256 锁定;训练在提交包外完成,包内无训练循环");
  const rows = [
    ["模型", "架构", "训练数据", "关键训练参数", "来源"],
    ["YOLO nano 检测器", "Ultralytics YOLO nano · 35 类", "detection_balanced_v2 本地数据集", "imgsz1536 · 120 epochs · batch4", "自训(v2 热启动;HR-DONK 开源件仅作初始基线)"],
    ["WirePinUNet 双头", "base32 共享 decoder(wire+pin)", "源注释几何栅格化 + 人工复核掩码", "512/384 瓦片 · 40 epochs · pos_weight 8/20 · epoch32 入选", "自训(milesial 积木,GPL-3 溯源)"],
    ["JunctionCNN", "6 通道双视角 CNN", "1024 合成渲染 + 人工复核裁片×12", "30 epochs · 置信阈 0.95 · 4 旋转一致", "自训(默认管线关闭,留作增强)"],
    ["PP-OCRv6 OCR", "det/rec small + cls(ONNX)", "—(开源预训练)", "页级 2048px;pin 级 2/3/4× × 0/90°", "直接调用(rapidocr 3.9.2)"],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
    fontSize: i === 0 ? 11 : 9.5, bold: i === 0, color: i === 0 ? WHITE : (j === 0 ? INK : MUTED),
    fill: { color: i === 0 ? PRIMARY : (i % 2 ? WHITE : "F6F9FC") }, fontFace: F, valign: "middle",
  } }))), { x: M, y: 1.4, w: 12.3, colW: [2.1, 2.5, 2.7, 2.6, 2.4], border: { pt: 0.75, color: "D5DFEA" }, rowH: 0.62 });
  s.addText("数据资产:1000 型号 EasyEDA 符号库 + 337 型号切片 + STM32F103C8T6 LQFP48 引脚库(附 ST 数据表 DS5319 出处);全部随镜像内置,运行零联网", { x: M, y: 5.28, w: 12.3, h: 0.4, fontSize: 10.5, color: INK, fontFace: F, margin: 0 });
  s.addText("溯源文件:weights/baseline_provenance.json(HR-DONK 初始基线,MIT)· vendor/pytorch_unet/provenance.json(milesial,GPL-3,逐文件 sha256)· 结点分类器训练数据含合成渲染器真值,无评测标注外泄", { x: M, y: 5.74, w: 12.3, h: 0.4, fontSize: 10, color: MUTED, fontFace: F, margin: 0 });
  footer(s, "口径:训练控制器在提交包外(Windows/GPU);weights/baseline_provenance.json 记录初始基线来源;vendor provenance 逐文件 sha256");
  pageNo(s, "B2");
}

/* ============ B3 · 运行流程与输出 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "备份 C · 核心代码与运行流程", "评测端 docker run 即用;开发可用已装依赖的 Python 直跑");
  // run commands
  card(s, 0.45, 1.35, 6.0, 2.6, "0D2237", "1E4D8F");
  s.addText([
    { text: "docker run --rm --gpus all \\\n", options: { color: "8FD8E2" } },
    { text: "  -v <输入>:/input:ro -v <输出>:/output:rw \\\n", options: { color: "8FD8E2" } },
    { text: "  pcb-parser:v102 -input /input -output /output\n\n", options: { color: "8FD8E2" } },
    { text: "# CPU 回退\n", options: { color: "6E86A6" } },
    { text: "python agent.py -input <dir> -output <dir> --device cpu\n", options: { color: "8FD8E2" } },
    { text: "# 约 30 个 PCB_* 环境开关,默认值即调参结果", options: { color: "6E86A6" } },
  ], { x: 0.7, y: 1.55, w: 5.6, h: 2.2, fontSize: 10, fontFace: MONO, margin: 0 });
  // pipeline chain
  chevronFlow(s, 0.45, 4.3, 12.4, 1.0, [
    ["run.sh", "镜像入口", PANEL],
    ["agent.py", "run_case 逐图", PANEL],
    ["S 阶段链", "S14→S24 子进程", PANEL],
    ["finalize ×15", "证据门控修复", PANEL],
    ["adapter", "target 校准", PANEL],
    ["final_results", "Pred.json", "E8F6F8"],
  ]);
  s.addText("冒烟复现产物(本机):decision_log.json + final_results/rp2350_page11_Pred.json + logs/(20+ 审计 JSON)已归档于 _audit_workspace/smoke/output_v102", { x: 0.5, y: 5.6, w: 12.3, h: 0.4, fontSize: 10.5, color: INK, fontFace: F, margin: 0 });
  s.addText("同名 PNG 冲突报错 · 支持子目录与前导零 · 部分失败写空 JSON 并继续 · 全部失败/预检失败返回非零", { x: 0.5, y: 6.05, w: 12.3, h: 0.4, fontSize: 10, color: MUTED, fontFace: F, margin: 0 });
  footer(s, "入口语义:镜像 ENTRYPOINT=/submission/run.sh(OCI config 已验证);/submission/agent_system 为官方占位,建议赛前与组委会确认调用方式");
  pageNo(s, "B3");
}

/* ============ B4 · 关键指标与消融 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "备份 D · 关键实验指标与消融(节选)", "全部数字可回查原始文件;逐行出处见 defense_supplement.md §三 统一数据表(17 行)");
  const rows = [
    ["实验", "数据范围", "指标", "结果"],
    ["检测配置选型", "59 例公开微平均", "检测 F1", "0.883(fused 胜出)"],
    ["元件引脚库映射", "13 页 released", "PinF1", "0.031 → 0.549(+47 号)"],
    ["未决结点网放行", "13 页 released", "连通率", "0.188 → 0.612 @0.83"],
    ["断线网放行", "13 页 released", "连通 / 总分", "0.569→0.576 / +0.31 分"],
    ["结点归并(案例)", "页 13", "wire TP/FP/FN · F1 · 总分", "32/28/17→46/13/3 · 58.72→85.19 · 82.871→92.656"],
    ["端子拓扑(案例)", "页 5", "连通 · 总分", "84.96%→100% · 93.954→99.474"],
    ["bbox canonical", "40 页公开子集", "component F1 · 达标率", ".8641→.8999 · 0.905→1.000"],
    ["bbox 按页缩放", "13 页 released", "20px 内模板元件", "150/163 → 159/163"],
    ["本机冒烟复现", "官方示例页", "输出规模 · bbox 对照", "89/165/63 · U3 差 1~2px"],
  ];
  s.addTable(rows.map((r, i) => r.map((c, j) => ({ text: c, options: {
    fontSize: i === 0 ? 11 : 9.5, bold: i === 0 || j === 3, color: i === 0 ? WHITE : (j === 3 ? PRIMARY : INK),
    fill: { color: i === 0 ? PRIMARY : (i % 2 ? WHITE : "F6F9FC") }, fontFace: F, valign: "middle", align: j === 3 ? "right" : "left",
  } }))), { x: M, y: 1.4, w: 12.3, colW: [2.6, 2.3, 2.9, 4.5], border: { pt: 0.75, color: "D5DFEA" }, rowH: 0.48 });
  footer(s, "口径警示:canonical(40 页)与页缩放(13 页)是两套独立实验;断线网“+0.31”是总分增量,连通率增量为 +0.7pp");
  pageNo(s, "B4");
}

/* ============ B5 · 评委问答速览 ============ */
{
  const s = pres.addSlide();
  s.background = { color: WHITE };
  header(s, "备份 E · 评委问答速览", "完整版见《EDA决赛答辩_QA.md》(10 问,含回答依据与出处)");
  const qas = [
    ["为什么不用大模型端到端?", "95% 权重在像素级结构精度;LLM 输出坐标不可靠且 Token 越少分越高(我们 =0)"],
    ["官方 74 与本地 98.93 的差距?", "口径不同:官方含 2/3 黑盒且难例 60%;本地仅 13 页公开集回放"],
    ["T 型十字怎么处理?", "保守拆分 + 红点检测;学习型分类器留作增强(v2 pilot,默认关)"],
    ["断线/漏检怎么办?", "gap 记录不乱接;修复阶段凭 60px 共线重合或红点+95% 墨迹恢复(+0.31 分)"],
    ["pin 编号读不到怎么办?", "条带 OCR(框外 30-50px)+ 密行重读 + 6 视角定向 OCR + 1000 型号库(PinF1 0.031→0.549)"],
    ["2 小时跑得完吗?", "GPU 3 并发设计;单例 wall_seconds 全程记录;官方机时未实测(如实回答)"],
  ];
  qas.forEach((q, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.45 + col * 6.3, y = 1.4 + row * 1.62;
    card(s, x, y, 6.1, 1.44);
    s.addShape(pres.shapes.OVAL, { x: x + 0.16, y: y + 0.18, w: 0.3, h: 0.3, fill: { color: PRIMARY } });
    s.addText("Q", { x: x + 0.16, y: y + 0.18, w: 0.3, h: 0.3, fontSize: 11, bold: true, color: WHITE, align: "center", fontFace: F, margin: 0 });
    s.addText(q[0], { x: x + 0.6, y: y + 0.12, w: 5.3, h: 0.4, fontSize: 12, bold: true, color: INK, fontFace: F, margin: 0 });
    s.addText(q[1], { x: x + 0.6, y: y + 0.56, w: 5.3, h: 0.8, fontSize: 9.5, color: MUTED, fontFace: F, margin: 0 });
  });
  footer(s, "所有回答依据均可指到 source_audit.md / defense_supplement.md 对应行");
  pageNo(s, "B5");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("WROTE", OUT));
