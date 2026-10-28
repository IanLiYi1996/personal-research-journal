# DetectorScope 深度调研

核查日期：2026-10-28。官方源码固定为 `52ecc87775154b35d4550911de22faad4fc68f40`。

## 阅读入口

- [技术原理与个人利用指南](/research-notes/2026-10-28-detectorscope.md)：从 SAE、JumpReLU、概念瓶颈到贡献计算、验证边界和试点路线。
- [论文证据核查](notes/paper-evidence.md)：原始论文版本、数据与指标口径。
- [源码证据核查](notes/source-audit.md)：实现位置、纯采样检查及未复现范围。

## 可重复的离线检查

```bash
python3 research-notes/2026-10-28-detectorscope/checks/principle_checks.py
```

从 personal-research-journal 仓库根目录执行。输出记录见 [results.json](checks/results.json)。

`source/` 只保留离线采样检查需要的固定版本 `detectorscope/autointerpret.py` 与官方 `LICENSE`，不包含完整第三方仓库、模型或数据。检查只提取已审阅的纯采样方法，不导入该文件的运行依赖，也不调用 Gemini。其余实现通过正文中的固定版本链接查阅。

没有执行训练／检测程序、下载模型或调用收费 API。研究笔记的入库与推送记录以 Git 历史为准。
