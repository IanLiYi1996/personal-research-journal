# 量子计算技术全景总结（索引）

- **Date:** 2026-04-15
- **Tags:** quantum-computing, survey, index

## Context

本文件为量子计算专题的**索引和概要**，每个主题的完整详细内容见对应的专题笔记。全系列基于 80+ 篇 arXiv 论文、Nature/Science 发表论文及官方公告撰写，所有内容均有明确来源引用。

| 专题笔记 | 内容 | 核心来源 |
| --- | --- | --- |
| [量子计算基础原理](2026-04-15-quantum-fundamentals.md) | Qubit/叠加/纠缠/量子门/电路模型/测量/退相干 | Mahmud & Goldsmith 2025; Zhao et al. 2024 |
| [量子硬件路线](2026-04-15-quantum-hardware.md) | 超导/离子阱/拓扑/光量子/中性原子/退火/对比 | Wong 2026; Haeffner et al. 2008; Henriet et al. 2020; Gill et al. 2024 等 14 篇 |
| [量子算法与应用](2026-04-15-quantum-algorithms.md) | Shor/Grover/VQE/QAOA/量子模拟/QML | Wang et al. 2026; Zhang et al. 2025; Aaronson et al. 2021 等 9 篇 |
| [量子纠错与容错计算](2026-04-15-quantum-error-correction.md) | Surface Code/LDPC/逻辑量子比特/Willow/资源估算 | Google Quantum AI 2024; Gidney 2025; Panteleev & Kalachev 2021 等 11 篇 |
| [量子计算行业格局](2026-04-15-quantum-industry-landscape.md) | Google/IBM/Microsoft/Amazon/Nvidia/IonQ/Quantinuum/D-Wave/中科大 | Arute et al. 2019; Acharya et al. 2024; IBM 2025 路线图等 38 篇 |
| [量子计算生态](2026-04-15-quantum-ecosystem.md) | QuEra/Atom Computing/PsiQuantum/Intel/Pasqal/国家战略/投资 | Wikipedia; quantum.gov; gov.uk; QURECA 等 20 篇 |
| [量子计算前沿与趋势](2026-04-15-quantum-frontiers.md) | 量子优势/混合计算/量子网络/后量子密码/QML | Wang et al. 2026; NIST PQC 2024; Masta et al. 2026 等 22 篇 |

---

## 技术体系总览

```text
一、基础原理
  ├── 量子比特 (Qubit) 与 Bloch 球
  ├── 叠加 (Superposition) / 纠缠 (Entanglement) / 干涉 (Interference)
  ├── 量子门 (Pauli-X/Y/Z, Hadamard, CNOT, Toffoli, Rotation)
  ├── 量子电路模型
  └── 测量、Born 规则、退相干 (T1/T2)

二、硬件路线
  ├── 超导量子比特 (Transmon) —— IBM, Google, USTC
  ├── 离子阱 (Trapped Ions) —— IonQ, Quantinuum
  ├── 拓扑量子比特 (Majorana) —— Microsoft
  ├── 光量子 (Photonic) —— Xanadu, PsiQuantum, USTC (九章)
  ├── 中性原子 (Neutral Atoms) —— QuEra, Pasqal
  └── 量子退火 (Quantum Annealing) —— D-Wave

三、核心算法
  ├── Shor 算法（整数分解，指数加速）
  ├── Grover 算法（无结构搜索，二次加速）
  ├── VQE / QAOA（变分量子算法，NISQ 时代核心）
  ├── 量子模拟（Hamiltonian 模拟，化学/材料）
  └── 量子机器学习 (QML)

四、量子纠错
  ├── 稳定子码 (Stabilizer Codes) 框架
  ├── 表面码 (Surface Code)：阈值 ~1%
  ├── qLDPC 码：渐近好码（Panteleev-Kalachev 2021）
  ├── Google Willow：首次低于阈值运行
  └── 资源估算：RSA-2048 从 10 亿→2000 万→<100 万物理量子比特

五、行业格局
  ├── 大厂：Google / IBM / Microsoft / Amazon / Nvidia
  ├── 商业公司：IonQ / Quantinuum / D-Wave
  └── 学术机构：USTC (九章/祖冲之) / MIT / Caltech

六、生态与战略
  ├── 新兴公司：QuEra / Atom Computing / PsiQuantum / Intel / Pasqal / Alice&Bob / Rigetti
  ├── 国家战略：美国 NQI / 中国 / EU Flagship / 英国 / 德法日韩
  ├── 投资格局：2025-2026 重大融资轮次
  └── DARPA 项目：US2QC / Quantum Benchmarking Initiative

七、前沿趋势
  ├── NISQ → Early FT → Full FT 过渡
  ├── 量子优势实验与经典模拟竞赛
  ├── 后量子密码学 (NIST FIPS 203/204/205)
  ├── 量子网络与分布式量子计算
  └── QML：非幺正方法、Fisher 效率跃迁
```

---

## 关键时间线

| 年份 | 里程碑事件 | 机构 |
|------|-----------|------|
| 1982 | Feynman 提出用量子系统模拟量子物理 | Caltech |
| 1994 | Shor 发表多项式时间整数分解量子算法 | Bell Labs |
| 1996 | Grover 发表量子搜索算法（二次加速） | Bell Labs |
| 1997 | Gottesman 提出稳定子码框架 | Caltech |
| 2011 | D-Wave One：首台商用量子退火计算机 (128 qubits) | D-Wave |
| 2019 | Google Sycamore 53 qubits 宣称量子优越性 | Google |
| 2020 | USTC 九章 1.0：76 光子光量子优越性 | 中科大 |
| 2021 | Panteleev-Kalachev 证明渐近好 qLDPC 码存在性 | — |
| 2023 | IBM Eagle 127 qubit "quantum utility" 证据 (Nature) | IBM |
| 2024 | Google Willow 105 qubits：首次低于阈值量子纠错 | Google |
| 2024 | USTC 祖冲之 3.0：105 qubit 新基准 | 中科大 |
| 2024 | NIST 发布三项后量子密码标准 (FIPS 203/204/205) | NIST |
| 2025 | Microsoft Majorana 1：首款拓扑核心处理器 (Nature) | Microsoft |
| 2025 | Amazon Ocelot：首款 Cat Qubit 纠错芯片 | Amazon |
| 2025 | Gidney：RSA-2048 分解降至 <100 万噪声量子比特 | Google |
| 2025 | D-Wave 量子退火优势 (Science) | D-Wave |
| 2026 | Nvidia Ising：全球首个开源量子 AI 模型 | Nvidia |

---

## 阅读指南

**如果你是量子计算新手：**
1. 先读 [基础原理](2026-04-15-quantum-fundamentals.md) 建立概念基础
2. 再读 [硬件路线](2026-04-15-quantum-hardware.md) 了解物理实现
3. 然后 [算法与应用](2026-04-15-quantum-algorithms.md) 了解量子计算能做什么
4. 最后 [行业格局](2026-04-15-quantum-industry-landscape.md) 看看谁在做什么

**如果你有计算机科学背景：**
1. 快速浏览 [基础原理](2026-04-15-quantum-fundamentals.md) 的数学部分
2. 直接进入 [算法与应用](2026-04-15-quantum-algorithms.md)（复杂度视角）
3. 深入 [纠错与容错](2026-04-15-quantum-error-correction.md)（这是当前最活跃的研究方向）
4. 阅读 [前沿趋势](2026-04-15-quantum-frontiers.md) 了解开放问题

**如果你关注产业动态：**
1. 直接读 [行业格局](2026-04-15-quantum-industry-landscape.md)
2. 补充 [前沿趋势](2026-04-15-quantum-frontiers.md) 的后量子密码和 QML 部分
3. 按需查阅技术细节
