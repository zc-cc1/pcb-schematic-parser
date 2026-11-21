# 答辩 PPT 模板来源与使用说明

调研日期:2026-11-21 · 调研方式:官方渠道在线核实(WebFetch 实际访问模板详情页)

## 一、候选模板比较(均来自 SlidesCarnival 正规渠道,免费)

| # | 模板 | 链接 | 配色/风格 | 规模 | 格式 | 许可 |
|---|---|---|---|---|---|---|
| 1 | **Dark Modern Tech Startup Brand Slides** | https://www.slidescarnival.com/template/tech-startup-brand/154090 | 深色/黑色现代科技,3D 元素,Oswald + Titillium Web | 22 页,16:9 | PPT/Google Slides/Canva | Free to use — no signups, no limits |
| 2 | **Bold Computer Engineering Technology** | https://www.slidescarnival.com/template/computer-engineering-technology/180403 | 黑+紫,工程科技风,几何感,单字体 Arimo | 20 页,16:9 | PPT/Google Slides/Canva | 同上 |
| 3 | **Futuristic Tech Pitch Deck** | https://www.slidescarnival.com/template/futuristic-tech-pitch-deck/55703 | 蓝+黑,未来感霓虹科技,Ubuntu 字体,数据可视化导向 | 25+ 页,16:9 | Canva 优先(可导出 PPT) | 同上 |

参考范围补充:Slidesgo(免费注册,每月 3 套免费模板,https://slidesgo.com/)、Microsoft Create、Canva 均核实可访问;Slidesgo 当期首页以节日/商业主题为主,深蓝工程风候选不足,故未纳入终选。

**许可条款出处**:SlidesCarnival 官方条款页 https://www.slidescarnival.com/terms-of-use#templates-license("Templates License":免费使用、无注册/下载限制;具体商用/再分发细则以该页为准)。本 PPT 为教育竞赛用途,符合许可范围。

## 二、选型结论与理由

**采用方案:Candidate 3 的"深蓝未来科技风"配色方向 + Candidate 1 的深色封面/浅色内页三明治结构,用 pptxgenjs 原生重建整套版式。**

理由:
1. **赛题气质匹配**:深蓝+青色是 EDA/芯片视觉领域的主导色系(Candidate 3 的蓝黑科技风最接近;Candidate 2 的紫色偏离需求,Candidate 1 纯黑偏消费创业)。
2. **可编辑性优先**:正式 PPT 需要后续微调文字/换图。直接下载的模板版式与本页内容结构(架构图、真实实验图、修复案例对照)差异大,逐页套模板会牺牲内容可读性——与任务要求"不要为了使用模板而牺牲内容可读性"一致,故选择以模板的设计语言为基准、原生重建。
3. **数据完整性**:评分权重图、指标对比等用 PowerPoint 原生图表/形状绘制,评委可直接读数,不受模板图表占位限制。

## 三、本 PPT 设计规范(即所选模板语言的具体化)

- 画布:16:9(13.333 × 7.5 in)
- 配色(全局常量):深蓝主底 `0B1F3A`(封面/结尾页)、内容页白底 `FFFFFF` + 面板淡蓝 `EEF4FA`;主色海军蓝 `123C6D`;辅色青 `0FB5C9`;强调橙 `F59E0B`(仅关键数字);正文深灰 `17263B`、弱化灰 `5A6B80`
- 字体:微软雅黑(Windows/Office 均内置;标题 Bold,正文 Regular;不使用细字重)
- 视觉母题:PCB 走线/焊盘纹样(封面与结尾页)、卡片式信息块、大数字 callout
- 页码统一右下角;来源口径行 12pt 灰字置底
- 所有实验图保持原始纵横比,按高度等比缩放放置,禁止拉伸
