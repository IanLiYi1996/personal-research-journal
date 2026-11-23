# DetectorScope 深度调研

核查日期：2026-10-28。官方源码固定为 `52ecc87775154b35d4550911de22faad4fc68f40`。

## 阅读入口

- [技术原理与个人利用指南](../2026-10-28-detectorscope.md)：从 SAE、JumpReLU、概念瓶颈到贡献计算、验证边界和试点路线。
- [论文证据核查](notes/paper-evidence.md)：原始论文版本、数据与指标口径。
- [源码证据核查](notes/source-audit.md)：实现位置、纯采样检查及未复现范围。

## 可重复的离线检查

```bash
python3 reports/detectorscope-2026-10-28/checks/principle_checks.py
```

从 `<workdir>` 执行。输出记录见 results.json。

`source/` 是官方仓库的固定版本副本，保留原许可。没有执行其训练／检测程序、下载模型、调用 Gemini 或收费 API。本次未向个人 journal 仓库提交或推送。
