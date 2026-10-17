# ClassicsClean · 古籍扫描件白底清洁工具

把泛黄、霉斑、背面透字严重的线装书扫描 PDF，转成**纸面接近纯白、笔画一个灰阶都不动**的
连续色调灰度 PDF。

- 输入：扫描版 PDF（内嵌彩色/灰度图像，可带馆藏章水印层）
- 输出：8bit 灰度 + JPEG2000 的 PDF，页面尺寸与源文档逐页一致
- 交付形态：`ClassicsClean.exe` —— 单文件，双击即用，也可命令行批处理

---

## 1. 教程

### 用 exe（推荐，目标机无需任何环境）

1. 双击 `ClassicsClean.exe`
2. 把 PDF **拖进**窗口虚线框，或点「选择 PDF 文件…」
3. 点「开始处理」
4. 成品落在源文件同目录，文件名追加 `_清洁版.pdf`

命令行批处理（不弹窗，适合批量）：

```bat
ClassicsClean.exe D:\books\a.pdf
ClassicsClean.exe a.pdf b.pdf -o D:\out --validate --report result.json
```

### 用源码

```bat
pip install -r requirements.txt
python -m classicsclean                       :: 启动图形界面
python -m classicsclean in.pdf -o out.pdf     :: 命令行
pytest                                        :: 单元测试（166 项，约 40 秒）
pytest -m slow                                :: 全书终检（真书 48 页，分钟级）
```

---

## 2. 目录结构

```
ClassicsClean/
├─ dist/ClassicsClean.exe   ★ 成品 exe（Win10/Win11 x64 通用，单文件免安装，约 36 MB）
│
├─ src/classicsclean/       ★ 源码（唯一来源，16 个模块，扁平分层见下）
├─ tests/                   ★ 测试代码（单元测试 + morphology 对照 + 全书终检 + 夹具）
├─ samples/                 ★ 测试资料（源 PDF）与成品 PDF
├─ scripts/                 ★ 打包与验收脚本（ClassicsClean.spec + build_exe.bat + verify_exe.py）
├─ docs/architecture.md     ★ 架构与算法深水区说明
│
├─ README.md                本文件：架构 + 用法 + 算法
├─ requirements.txt         运行依赖（纯 numpy 形态学，不含 OpenCV）
├─ requirements-dev.txt     开发 / 测试 / 打包依赖
├─ pyproject.toml           项目元数据与依赖声明
└─ pytest.ini               测试配置（默认跳过分钟级的 slow 终检）
```

**为什么这样分**：源码、测试、样本、脚本四个目录各对应一类「人要看的东西」——
`src/` 看代码、`tests/` 看测试、`samples/` 看效果、`scripts/` 看打包；
`docs/` 放架构与算法的深水区说明，成品落在 `dist/`。

---

## 3. 代码架构

```
第 4 层  入口   __main__.py ──┬── cli.py   命令行（批处理 / 无头）
                               └── gui.py   图形界面（拖拽 / 选择文件）
                                      │  两者都只调用第 3 层，不重复任何算法逻辑
第 3 层  编排   process_pdf.py   整本：读源 PDF → 逐页 → 写 PDF → 回读校验
               process_page.py  单页：把第 2 层算法串成一条流水线
                                      │
第 2 层  算法   masks.py        掩膜分割：白边 / 墨迹 / 破洞 / 纸面
               white_point.py  白场估计：最白区 · 局部白场 · 兜底
               tone.py         色调映射 + 文字保护（红线所在）
               repair.py       破洞修补（碰字的洞一律放过）
                                      │
第 1 层  基础   imaging.py      PDF ↔ 位图、灰度化、JPEG2000 编解码
               morphology.py   形态学 / 滤波 / 重采样原语（纯 numpy，无 OpenCV）
               config.py       参数唯一来源（所有常量集中在此）
```

单页数据流：

```
源页面位图 ─┬─→ masks.compute_masks ──→ 白边 / 墨迹 / 破洞 / 纸面（互斥）
            ├─→ white_point ─────────→ 基线白点 Wp + 逐像素白场 Wp_map
            ├─→ tone.tone_map ───────→ 亮度拉平 + gamma 高光压缩（纸面推向白）
            ├─→ tone.text_guard ─────→ 笔画回落到「基线白场 + 线性映射」（字不动）
            └─→ repair.repair_holes ─→ 补不涉及文字的破洞
                     ↓
          输出灰度图 ─→ imaging.encode_jp2 ─→ 成品 PDF 页面
```

**分层原则**：第 1、2 层不接触文件也不接触界面，只吃 numpy 数组、吐 numpy 数组；
第 3 层 `process_pdf.py` 是全项目唯一的 IO 边界；第 4 层只是同一套算法的两种入口。

---

## 4. 四条红线（不可违反）

| # | 红线 | 守护者 |
|---|------|--------|
| 1 | 纯白扫描边输出恒为 255 | `tone.tone_map` |
| 2 | 笔画及抗锯齿边缘**逐像素**等于「基线白场 + 线性映射」，差必须为 0 | `tone.text_guard_weight` |
| 3 | 白场只统计纸面，与白边 / 墨迹 / 破洞的重叠像素恒为 0 | `white_point.whitest_region` |
| 4 | 碰字的破洞一律不补 | `repair.repair_holes` |

---

## 5. 算法要点

1. **只取内嵌主图像**：扫描古籍常在图像上叠加 OCG 馆藏章水印层，整页渲染会把水印混进来。
2. **四掩膜互斥**：白边（与边缘连通的亮区）、墨迹（< 0.62×纸面中位）、破洞（内部孤立亮块）、纸面。
   所有统计只在纸面上做。
3. **局部白场**（定稿关键）：线装纸的黄变/霉斑是**空间非均匀**的——实测约 45% 纸面像素比全局最白区暗 12 灰阶以上，
   单一全局白点在数学上不可能同时把亮区和暗区推到白。改用多尺度（31/61/121）局部纸面亮度场作逐点白点。
4. **gamma 3.0 高光压缩**：只调整映射曲线，白点规则保持不变，把残留底灰推向纯白。
5. **文字硬保护**：笔画 + 抗锯齿边缘权重恒为 1，输出直接沿用基线白场的线性映射（实测差 = 0）。
   木版印刷的洇墨晕边比纸面暗得不多，会被 gamma 漂白成「文字偏浅」，因此保护掩膜
   还会把「比局部纸面暗超过 15 灰阶」的晕边像素一并纳入（`text_dev_thr`）。
6. **JPEG2000 52 dB**：45 dB 时字核心编码噪声达 42 灰阶，50 dB→10、52 dB→16 以内，所以定 52 dB。

所有常量集中在 `src/classicsclean/config.py`，改一个数字就能整体调整效果，
改动前请先看该文件里每个字段的注释。

---

## 6. 测试与验收

| 层次 | 位置 | 跑法 | 耗时 |
|------|------|------|------|
| 单元测试（167 项） | `tests/unit/test_*.py` | `pytest` | ~60 s |
| 全书终检（真书 48 页） | `tests/acceptance/` | `pytest -m slow` | 分钟级 |
| 污渍/字迹平衡检测 | `scripts/run_balance.py` | `python scripts/run_balance.py` | 分钟级 |

结论摘要与逐页数据见 `reports/` 下的 `acceptance.json`、`acceptance-pages.csv`、`balance.json`。

---

## 7. 兼容性

| 项目 | 说明 |
|------|------|
| 操作系统 | Windows 10 / 11，x64（单文件 exe，目标机无需 Python） |
| 运行时依赖 | 已全部打包进 exe（含 VC++ 运行库、MuPDF/qpdf/OpenJPEG）；**不含 OpenCV** —— 形态学 / 滤波 / 重采样由纯 numpy 的 `morphology.py` 实现，与 cv2 逐像素等价 |
| 体积 | 约 36 MB（纯 numpy 替代 OpenCV + 去掉未用插件 + UPX LZMA 压缩，原版 99 MB） |
| 输入格式 | 扫描图像页 PDF；纯文本 / 矢量页会退化为 200 dpi 整页渲染 |
| 输出格式 | PDF，8bit DeviceGray + JPXDecode（JPEG2000） |
