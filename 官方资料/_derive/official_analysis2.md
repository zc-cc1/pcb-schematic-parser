# 官方资料分析（二）：真实型号覆盖与引脚对齐

## 1. `open_GT`（13 例）—— 目前唯一 Name=芯片型号 的标注集

| case | 元件 | 引脚 | 网络 | 图片尺寸 |
|---|---:|---:|---:|---|
| 1 | 58 | 135 | 40 | 2240×1149 |
| 2 | 28 | 48 | 24 | 2063×1116 |
| 3 | 26 | 45 | 16 | 2247×1195 |
| 4 | 24 | 43 | 14 | 1849×1032 |
| 5 | 26 | 49 | 17 | 2057×1096 |
| 6 | 38 | 68 | 22 | 2186×1110 |
| 7 | 24 | 48 | 13 | 2202×1003 |
| 8 | 20 | 45 | 18 | 2229×1192 |
| 9 | 17 | 31 | 11 | 2222×1199 |
| 10 | 29 | 57 | 15 | 2200×1017 |
| 11 | 15 | 41 | 14 | 1469×770 |
| 12 | 21 | 38 | 16 | 1658×752 |
| 13 | 20 | 44 | 11 | 1370×797 |

合计：346 元件 / 346 引脚 / 231 网络

## 2. 四份资料里 `type` 的写法不一致（这是 ComponentF1 的隐性扣分点）

| 资料 | 类型数 | 取值示例 |
|---|---:|---|
| open_GT(13) | 11 | `gnd`, `c`, `r`, `v`, `box`, `led`, `l`, `net_bidirection`, `d`, `switch`, `pin` |
| 10_GTcase(10) | 20 | `r`, `gnd`, `c`, `net_input`, `box`, `v`, `l`, `mosfet_npn`, `pin`, `d`, `net_output`, `other` |
| 200_train(200) | 41 | `gnd`, `c`, `r`, `v`, `box`, `net_input`, `d`, `led`, `pin`, `net_short`, `switch`, `l` |
| case_001扩展示例 | 4 | `vbus_in`, `resistor`, `ic`, `gnd` |

- 三份数据共有的 type 只有 10 个：`box`, `c`, `d`, `gnd`, `l`, `led`, `pin`, `r`, `switch`, `v`
- open_GT 与 10GT 的并集 21 类，
  再与 200 训练集并集 43 类。
  官方 Q&A 说 `type_list` 是 34 类且评测不会超出——**该文件仍未出现在本文件夹**，
  但上面这个并集给出了候选范围；`ic`/`resistor`/`vbus_in` 这类长名只出现在 `case_001扩展示例`，
  与 10GT 的短枚举（`box`/`r`）互斥，混用会直接判错 type。

## 3. 真实型号（Name 字段）在三库中的可查率

### open_GT(13) 的 box/ic 元件（31 个）

| case | key | Name | 命中库 | value |
|---|---|---|---|---|
| 1 | U1 | `IP5306-4.35V` | BoxICLibrary.json | None |
| 1 | U2 | `SX1308` | BoxICLibrary.json | None |
| 1 | U3 | `XB7608A` | BoxICLibrary.json | None |
| 1 | USB1 | `5V2A in` | **无** | None |
| 1 | USB2 | `5V2A out` | **无** | None |
| 2 | U1 | `HS96L03W2C03` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 2 | U2 | `HS96L03W2C03` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 2 | U3 | `HS96L03W2C03` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 2 | U4 | `HS96L03W2C03` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 3 | U1 | `SY8113BADC` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 4 | U1 | `XC6206P332MR` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 4 | U2 | `AMS1117-3.3` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 4 | U3 | `TP4054-MS` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 5 | U1 | `TPS61240DRVR` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 5 | U2 | `TPS61240DRVR` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 5 | U3 | `AMS1117-3.3` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 6 | U1 | `78M12` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 6 | U2 | `STMPS2141STR` | BoxICLibrary.json | None |
| 6 | U3 | `78M12` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 6 | U4 | `STMPS2141STR` | BoxICLibrary.json | None |
| 7 | U1 | `MP2236CGJ-Z` | BoxICLibrary.json | None |
| 8 | U1 | `PAW3395DM-T6QU` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 9 | U2 | `MT2492` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 10 | U1 | `EG1192L` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 10 | U2 | `CJ78L05` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 11 | U1 | `TP4333` | BoxICLibrary.json | None |
| 11 | USB1 | `TYPE-C 6P(073)` | BoxICLibrary.json / BoxICLibrary_200cases.json / cpntLibrary | None |
| 11 | USB2 | `AF 90° WJDG` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 12 | U1 | `SY6912AFCC` | BoxICLibrary.json | None |
| 13 | U1 | `RT7272BGSP` | BoxICLibrary.json | None |
| 13 | H1 | `H1` | **无** | None |

可查率：28/31

### 10_GTcase(10) 的 box/ic 元件（15 个）

| case | key | Name | 命中库 | value |
|---|---|---|---|---|
| 0030 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0030 | U2 | `U2` | **无** | None |
| 0036 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0042 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0059 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0065 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0070 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0070 | U2 | `U2` | **无** | None |
| 0099 | IR4428 | `IR4428` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0099 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0099 | U2 | `U2` | **无** | None |
| 0100 | U1 | `U1` | BoxICLibrary.json / BoxICLibrary_200cases.json | None |
| 0100 | U2 | `U2` | **无** | None |
| 0100 | U3 | `U3` | **无** | None |
| 0100 | U4 | `U4` | **无** | None |

可查率：9/15

## 4. 库里的引脚与 GT 对齐情况（Name 能查到的 box 元件）

| case | key | Name | GT引脚 | 库引脚 | 编号完全一致 | 共同编号 | 名称完全一致 | 库中 `x_y` 形式编号 |
|---|---|---|---:|---:|---|---|---|---|
| open_GT/1 | U1 | `IP5306-4.35V` | 9 | 9 | Y | 9 | Y | 0 |
| open_GT/1 | U2 | `SX1308` | 6 | 6 | Y | 6 | Y | 0 |
| open_GT/1 | U3 | `XB7608A` | 5 | 5 | Y | 5 | Y | 0 |
| open_GT/2 | U1 | `HS96L03W2C03` | 4 | 4 | Y | 4 | Y | 0 |
| open_GT/2 | U2 | `HS96L03W2C03` | 4 | 4 | Y | 4 | Y | 0 |
| open_GT/2 | U3 | `HS96L03W2C03` | 4 | 4 | Y | 4 | Y | 0 |
| open_GT/2 | U4 | `HS96L03W2C03` | 4 | 4 | Y | 4 | Y | 0 |
| open_GT/3 | U1 | `SY8113BADC` | 6 | 6 | Y | 6 | Y | 0 |
| open_GT/4 | U1 | `XC6206P332MR` | 3 | 3 | Y | 3 | Y | 0 |
| open_GT/4 | U2 | `AMS1117-3.3` | 4 | 4 | Y | 4 | Y | 0 |
| open_GT/4 | U3 | `TP4054-MS` | 5 | 5 | Y | 5 | Y | 0 |
| open_GT/5 | U1 | `TPS61240DRVR` | 7 | 7 | Y | 7 | Y | 0 |
| open_GT/5 | U2 | `TPS61240DRVR` | 7 | 7 | Y | 7 | Y | 0 |
| open_GT/5 | U3 | `AMS1117-3.3` | 4 | 4 | Y | 4 | N | 0 |
| open_GT/6 | U1 | `78M12` | 3 | 3 | Y | 3 | Y | 0 |
| open_GT/6 | U2 | `STMPS2141STR` | 5 | 5 | Y | 5 | Y | 0 |
| open_GT/6 | U3 | `78M12` | 3 | 3 | Y | 3 | Y | 0 |
| open_GT/6 | U4 | `STMPS2141STR` | 5 | 5 | Y | 5 | Y | 0 |
| open_GT/7 | U1 | `MP2236CGJ-Z` | 8 | 8 | Y | 8 | Y | 0 |
| open_GT/8 | U1 | `PAW3395DM-T6QU` | 16 | 16 | Y | 16 | Y | 0 |
| open_GT/9 | U2 | `MT2492` | 6 | 6 | Y | 6 | Y | 0 |
| open_GT/10 | U1 | `EG1192L` | 9 | 9 | Y | 9 | Y | 0 |
| open_GT/10 | U2 | `CJ78L05` | 3 | 3 | Y | 3 | Y | 0 |
| open_GT/11 | U1 | `TP4333` | 8 | 8 | Y | 8 | Y | 0 |
| open_GT/11 | USB1 | `TYPE-C 6P(073)` | 7 | 7 | Y | 7 | Y | 0 |
| open_GT/11 | USB2 | `AF 90° WJDG` | 5 | 5 | Y | 5 | Y | 0 |
| open_GT/12 | U1 | `SY6912AFCC` | 9 | 9 | Y | 9 | Y | 0 |
| open_GT/13 | U1 | `RT7272BGSP` | 9 | 9 | Y | 9 | Y | 0 |
| 10GT/0030 | U1 | `U1` | 3 | 30 | N | 1 | N | 0 |
| 10GT/0036 | U1 | `U1` | 8 | 30 | N | 8 | N | 0 |
| 10GT/0042 | U1 | `U1` | 20 | 30 | N | 18 | N | 0 |
| 10GT/0059 | U1 | `U1` | 6 | 30 | N | 4 | N | 0 |
| 10GT/0065 | U1 | `U1` | 8 | 30 | N | 6 | N | 0 |
| 10GT/0070 | U1 | `U1` | 6 | 30 | N | 6 | N | 0 |
| 10GT/0099 | IR4428 | `IR4428` | 8 | 8 | N | 7 | Y | 0 |
| 10GT/0099 | U1 | `U1` | 3 | 30 | N | 0 | N | 0 |
| 10GT/0100 | U1 | `U1` | 28 | 30 | N | 28 | N | 0 |

编号集合**完全一致** 28/37；名称集合完全一致 28/37。

> 结论：库的 `pins` 是“多个同源符号实例”的聚合，条目数通常远多于物理引脚，
> 且重复编号带 `_1/_2` 后缀。**不能直接把库引脚当输出**；只能按编号/名称做交集匹配后回填。

## 5. 可回填性小结（对当前 PinF1/NetF1 的口径）

- 10GT 与 open_GT 的 `pinname` 覆盖率都是 100%，378 / 全量引脚都带名字 → pinname 是可比字段。
- 官方 Q&A 第 5/29 条：电源/地类 pin number 是**随机生成**，不参与评分；第 4 条：VCC/GND 直接按坐标匹配，
  同名 GND/VCC 在打分时**合并为同一 net**。
- 官方 Q&A 第 3 条（评测问题）：非 box 元件的 pinname/pinnumber **不参与评分**。
  → 库的价值集中在 box/ic 的引脚名与编号上，其余器件只需 bbox+type。
