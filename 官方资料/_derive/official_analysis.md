# 官方资料自动分析

## 1. 文件清单与重复

| 文件 | 大小 | SHA256 前12 | 与公开数据集对比 |
|---|---:|---|---|
| BoxICLibrary.json | 3,139,669 | b6f7fd065615 | - |
| BoxICLibrary_200cases.json | 1,826,150 | d5e0b346eb87 | - |
| README.md | 4,353 | 1858ce1d4419 | - |
| _json_peek.txt | 6,265 | 50551ed0044d | - |
| cpntLibrary | 1,755,587 | 9c2bec334d41 | 相同 |
| cpntLibrary.md | 3,174 | 051169b86fae | - |
| 赛题Q&A1013.docx | 955,583 | a1017c46b72f | - |

## 2. 三个元器件库的关系

| 库 | 条目数 | 每条字段 |
|---|---:|---|
| BoxICLibrary.json | 1000 | pins x1000 |
| BoxICLibrary_200cases.json | 360 | esym_path/pins/same_hash_components x241; esym_path/pins x119 |
| cpntLibrary | 337 | esym_path/pins/same_hash_components x337 |

| 组合 | 条目数 |
|---|---:|
| BoxICLibrary_200cases ∩ cpntLibrary | 145 |
| BoxICLibrary_200cases − cpntLibrary | 215 |
| cpntLibrary − BoxICLibrary_200cases | 192 |
| BoxICLibrary ∩ BoxICLibrary_200cases | 360 |
| BoxICLibrary − BoxICLibrary_200cases | 640 |

`BoxICLibrary_200cases − cpntLibrary` 示例：24C02C, 25MHz, 4C4001, 4N28, 555, 7404, 74LS154, 7805, 78L05, 78M05, 79M12, 8051, 8MHz, 93LC56BT-I/OT, A, A111 Aconner, ADUM1411BRWZ, ADV7180BCPZ-REEL, AF 90° WJDG, AFA07-S12FCA-00

`cpntLibrary − BoxICLibrary_200cases` 示例：1.8 TFT 128*RGB*160, 3.3V, 78M12_C116325, 78M12_C499638, ABS210_C123897, ABS210_C224221, ABS210_C2923955, ABS210_C525748, AMS1117-3.3_1, AMS1117-3.3_1839, AMS1117-3.3_2, AMS1117-3.3_C165482, AMS1117-3.3_C18212698, AMS1117-3.3_C20628883, AMS1117-3.3_C2891839, AMS1117-3.3_C2904742, AMS1117-3.3_C2992569, AMS1117-3.3_C2992570, AMS1117-3.3_C347222, AMS1117-3.3_C347256

| 库 | 引脚条目 | 形如 `1_1` 的编号 | 占比 |
|---|---:|---:|---:|
| BoxICLibrary.json | 24,708 | 138 | 0.6% |
| BoxICLibrary_200cases.json | 13,600 | 138 | 1.0% |
| cpntLibrary | 9,762 | 0 | 0.0% |

## 3. `type` 取值：10GT 与 200 训练集

- 10GT：20 类 —— `r`×41, `gnd`×34, `c`×25, `net_input`×16, `box`×15, `v`×10, `l`×6, `mosfet_npn`×6, `pin`×5, `d`×5, `net_output`×4, `other`×3, `battary`×3, `crystal_2pin`×3, `switch`×2, `bjt_npn`×2, `led`×1, `bjt_pnp`×1, `amp`×1, `circle_header`×1
- 200 训练集：41 类
- 训练集有、10GT 没有：['amp_3pin', 'amp_5pin', 'ant', 'bjt', 'block', 'buzzer', 'crystal', 'crystal_3pin', 'crystal_4pin', 'dc', 'esd', 'fuse', 'jumper', 'm3螺丝', 'mosfet', 'mosfet_pnp', 'motor', 'net_bidirection', 'net_short', 'opt', 'seg', 'testpoint', 'transfomer']
- 10GT 有、训练集没有：['bjt_npn', 'mosfet_npn']

> 官方 Q&A 第3/4/14/17条：以官方 `type_list` 为准（34 类），评测不会超出该表；
> 该文件**不在本文件夹中**，需要另外索取。10GT 是当前唯一能直接观测到的官方 `type` 样本。

## 4. 10GT 元件能否在库中查到（按 Name 精确匹配）

| case | 元件 | type | Name | value | 在 cpntLibrary | 在 BoxICLibrary_200cases | 在 BoxICLibrary |
|---|---|---|---|---|---|---|---|
| 0030 | U1 | box | U1 |  |  | Y | Y |
| 0030 | U2 | box | U2 |  |  |  |  |
| 0036 | U1 | box | U1 |  |  | Y | Y |
| 0042 | U1 | box | U1 |  |  | Y | Y |
| 0059 | U1 | box | U1 |  |  | Y | Y |
| 0065 | U1 | box | U1 |  |  | Y | Y |
| 0070 | U1 | box | U1 |  |  | Y | Y |
| 0070 | U2 | box | U2 |  |  |  |  |
| 0099 | IR4428 | box | IR4428 |  |  | Y | Y |
| 0099 | U1 | box | U1 |  |  | Y | Y |
| 0099 | U2 | box | U2 |  |  |  |  |
| 0100 | U1 | box | U1 |  |  | Y | Y |
| 0100 | U2 | box | U2 |  |  |  |  |
| 0100 | U3 | box | U3 |  |  |  |  |
| 0100 | U4 | box | U4 |  |  |  |  |

10GT 元件总数 184；Name 命中 cpntLibrary 的 0，命中 BoxICLibrary_200cases 的 9，命中 BoxICLibrary 的 9。

- `STMPS2141STR`：cpntLibrary=无，BoxICLibrary_200cases=无，BoxICLibrary=有
- `MP2236CGJ-Z`：cpntLibrary=无，BoxICLibrary_200cases=无，BoxICLibrary=有

## 5. box 类元件的引脚：库里查到的编号/名称 vs GT 实际

| case | 元件 | GT type | Name | GT引脚数 | 库引脚数 | 编号集合是否一致 | 名称集合是否一致 | 差异示例 |
|---|---|---|---|---:|---:|---|---|---|
| 0030 | U1 | box | U1 | 3 | 30 | N | Y | GT缺 ['10', '11', '12', '13'] / 库缺 ['39', 'FB'] |
| 0036 | U1 | box | U1 | 8 | 30 | N | Y | GT缺 ['10', '11', '12', '13'] / 库缺 [] |
| 0042 | U1 | box | U1 | 20 | 30 | N | Y | GT缺 ['13', '19', '21', '22'] / 库缺 ['19_1', '19_2'] |
| 0059 | U1 | box | U1 | 6 | 30 | N | N | GT缺 ['1', '10', '11', '12'] / 库缺 ['C1', 'C2'] |
| 0065 | U1 | box | U1 | 8 | 30 | N | N | GT缺 ['1', '10', '11', '12'] / 库缺 ['8_1', '8_2'] |
| 0070 | U1 | box | U1 | 6 | 30 | N | Y | GT缺 ['1', '10', '11', '13'] / 库缺 [] |
| 0099 | IR4428 | box | IR4428 | 8 | 8 | N | Y | GT缺 ['9'] / 库缺 ['6'] |
| 0099 | U1 | box | U1 | 3 | 30 | N | N | GT缺 ['1', '10', '11', '12'] / 库缺 ['154', '156', 'D2 BAT54S'] |
| 0100 | U1 | box | U1 | 28 | 30 | N | Y | GT缺 ['29', '30'] / 库缺 [] |

编号集合完全相同：0/9。库的 `number` 是同名符号多实例的**聚合**，
出现重复编号时会加 `_1/_2` 后缀（见第2节统计），直接当 `pin_<number>` 输出会造出 GT 里不存在的键。

## 6. 200 训练集里 GT 引脚名的可观测性（供 OCR/库回填评估）

- 200 训练集引脚总数 21,819，其中 `pinname` 非空 21,819（100.0%）
- 10GT 引脚总数 378，其中 `pinname` 非空 378（100.0%）
