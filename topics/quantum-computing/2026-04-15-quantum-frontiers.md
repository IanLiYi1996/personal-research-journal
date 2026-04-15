# 量子计算前沿与趋势

- **Date:** 2026-04-15
- **Tags:** quantum-computing, quantum-advantage, fault-tolerant, PQC, QML, frontiers

## Context

本文基于 2024-2026 年间的最新论文与进展，综述量子计算领域的关键前沿方向。主要参考文献包括：Google Willow 芯片的纠错突破 [1]、Microsoft 拓扑量子比特路线图 [2]、VQA 向容错过渡的综述 [3]、量子网络基础教程 [4]、量子化学应用展望 [5]、QML 核方法综述 [6]、非幺正量子机器学习实证研究 [7]、分布式量子优化框架 [8]，以及 NIST 后量子密码标准 [9] 等。

---

## 一、NISQ 到容错量子计算的过渡

量子计算当前正处于从 NISQ（Noisy Intermediate-Scale Quantum）向容错（Fault-Tolerant）时代过渡的关键阶段。这一过渡可分为三个递进阶段：NISQ、早期容错（Early Fault-Tolerant, EFT）和最终容错（Ultimate Fault-Tolerant, FT）。

**纠错里程碑：Google Willow 芯片。** 2024 年 12 月，Google 发布 Willow 量子芯片（105 个量子比特），首次在超导系统中实现 surface code 纠错阈值以下运行 [1]。实验表明，随着编码距离从 3×3 增大到 5×5 再到 7×7，逻辑错误率每次减半，实现了指数级错误抑制。距离-7 的 surface code 存储器达到每周期 0.143% ± 0.003% 的错误率，逻辑量子比特寿命超过最佳物理量子比特 2.4 ± 0.3 倍 [1]。这是量子纠错领域追求近 30 年的目标。

**拓扑量子比特路径。** Microsoft 提出了基于 Majorana 拓扑量子比特的四代设备路线图 [2]：（1）单量子比特设备——实现基于测量的基准测试；（2）双量子比特设备——通过测量编织执行 Clifford 门；（3）八量子比特设备——证明逻辑比特操作优于物理比特；（4）拓扑量子比特阵列——支持 lattice surgery 演示。该方案依赖超导体-半导体异质结构支撑拓扑相，代表了与超导量子比特完全不同的容错计算路径。

**VQA 的角色演变。** Wang et al. 的综述 [3] 指出，变分量子算法（VQA）已成为 NISQ 时代的核心计算范式。随着硬件向 EFT/FT 过渡，VQA 的基本原理需要系统性重新评估。在 EFT 阶段，VQA 将与量子纠错缓解（error mitigation）和部分纠错相结合；在 FT 阶段，纯混合式 VQA 模型面临根本性挑战，需要向全量子求解器演进 [3]。

![VQA 混合量子-经典架构：参数化量子电路 (Ansatz) 设计、测量与经典优化循环 [3]](2026-04-15-quantum-frontiers/fig-vqa-architecture.png)

---

## 二、量子优势实验：已证明了什么？

"量子优势"（Quantum Advantage）的定义和持久性至今仍存在争议。

**Google Sycamore（2019）。** 使用 53 量子比特处理器完成 random circuit sampling 任务，声称经典超级计算机需要约 10,000 年 [10]。IBM 随即质疑这一估计。截至 2024 年，Google 研究人员自身估计，经典张量网络算法的改进已使 53 量子比特模拟仅需约 6 秒 [10]。

**中国实验：九章与祖冲之。** 2020 年 12 月，光量子计算机九章通过 Gaussian boson sampling 检测到 76 个光子，声称经典模拟需 25 亿年 [10]。2021 年，九章 2.0 与祖冲之进一步推进，后者展示了比 Sycamore 困难 6 个数量级的采样任务 [10]。

**Google Willow（2024）。** Willow 芯片在不到 5 分钟内完成了一项基准计算，该计算在经典超级计算机上估计需要约 10^25 年 [1]。虽然这一声明令人印象深刻，但 random circuit sampling 的实际应用价值仍受质疑。

**关键启示。** 量子优势的证明是一场"移动的球门"——每当量子实验取得突破，经典算法也在快速改进 [10]。目前公认的共识是：在特定精心设计的任务上量子设备确实超越经典计算，但尚未在具有实际商业价值的问题上展示无可争辩的优势。

---

## 三、量子-经典混合计算

混合量子-经典（Hybrid Quantum-Classical）架构是当前 NISQ 时代最实用的计算范式。

**VQE 与 QAOA 的实践。** 变分量子本征求解器（VQE）在量子化学中取得进展。Alexeev et al. [5] 指出，拥有 25-100 个逻辑量子比特的早期容错量子计算机可在量子化学中实现实际影响——特别是多参考态电荷转移、锥交叉态等光化学与材料设计的核心问题。这些场景下量子计算可采用经典求解器难以实现的多项式量级相位估计、量子动力学直接模拟、活性空间嵌入等策略 [5]。

![量子化学模拟的战略方向：不同尺度下的量子系统与计算方法映射 [5]](2026-04-15-quantum-frontiers/fig-quantum-chemistry-directions.png)

![量子化学模拟路线图：随逻辑量子比特预算增长的能力演进（未来五年展望）[5]](2026-04-15-quantum-frontiers/fig-quantum-chemistry-roadmap.png)

**Barren Plateaus：核心瓶颈。** 梯度消失（barren plateaus）是 VQA 训练的核心障碍 [3]。Hirviniemi et al. [11] 通过加强 small-angle initialization 的理论证明，提出了一种免受 barren plateaus 影响的完整电路模型，同时对经典张量网络和 Pauli propagation 模拟技术具有鲁棒性，为 NISQ 设备上的量子生成模型提供了新可能。

**从混合到全量子。** Qiao et al. [12] 提出了一种全量子变分动力学模拟方法，通过 Chebyshev 谱离散化将时间演化的常微分方程转化为静态线性方程，利用量子奇异值变换求解。该方法避免了经典反馈，电路深度与时间步数无关，为从混合变分算法向全量子求解器的过渡建立了系统路径。

---

## 四、量子网络与分布式量子计算

量子网络是实现量子互联网（Quantum Internet）和分布式量子计算的基础设施。

**距离记录。** 2024 年，中国与南非研究人员通过低轨微卫星实现了 12,900 km 的大气量子密钥分发（QKD），在单次轨道中传输了超过 100 万个量子安全比特 [13]。Twin-Field QKD 实验已将光纤距离推至 833.8 km [13]。中国的京沪干线（2,000 km，32 个可信节点）于 2021 年扩展为跨越约 4,600 km 的全球最大综合量子通信网络 [14]。

**分布式量子计算进展。** 2025 年 2 月，牛津大学研究人员展示了两个光子连接的离子阱模块之间的分布式量子计算，controlled-Z 门远程传输保真度达 86%，成功执行 Grover 搜索算法（成功率 71%）[14]。

**架构挑战。** Gkelias et al. [4] 的教程指出，当前量子网络开发面临"仿真-现实鸿沟"（simulation-reality gap）——物理社区专注硬件，经典网络社区缺乏管理脆弱量子资源的架构模型。他们提出软件定义量子网络（SDQN）作为可扩展性的先决条件，建立了经典控制平面指导量子数据流的双平面架构 [4]。

**分布式量子优化。** Huang et al. [8] 提出基于因子图范式的结构感知分布式量子优化框架，在保持 Grover 级 O(√N) 查询复杂度的同时降低了每设备量子比特需求。该框架支持全相干模式（面向容错网络）和混合模式（限制电路深度以适应近期设备）[8]。

**量子中继器。** 量子中继器的实用化部署仍面临重大挑战 [13]。Swain & Ralph [15] 提出基于 GKP 玻色子码的量子中继器方案，可在室温下运行，利用 GKP 量子比特实现与光子量子比特方案相当的性能，同时所需量子比特数减少数个数量级。

---

## 五、后量子密码学（NIST 标准）

量子计算对现有公钥密码体系构成根本威胁。Shor 算法可在多项式时间内破解 RSA 和 ECC，"先收集、后解密"（Harvest-Now, Decrypt-Later, HNDL）攻击使这一威胁具有现实紧迫性。

**NIST 标准化里程碑。** 2024 年 8 月，NIST 正式发布三项后量子密码标准 [9]：
- **FIPS 203（ML-KEM）**：基于模格的密钥封装机制（原 CRYSTALS-Kyber），用于通用加密；
- **FIPS 204（ML-DSA）**：基于模格的数字签名算法（原 CRYSTALS-Dilithium），用于数字签名；
- **FIPS 205（SLH-DSA）**：无状态哈希签名算法（原 SPHINCS+），作为 ML-DSA 的备选方案。

此外，基于 FALCON 的 FIPS 206（FN-DSA）计划于 2024 年底发布 [9]。

**性能评估。** Chhetri [16] 在 ARM Cortex-M0+ 上进行了 ML-KEM 和 ML-DSA 的首次系统基准测试。ML-KEM-512 在 133 MHz 处理器上完成完整密钥交换仅需 35.7 ms，比同硬件上的 ECDH P-256 快 17 倍。在区块链场景中，ML-DSA 在安全级别 5 的验证时间为 0.14 ms，而 ECDSA 为 0.88 ms [17]。

**5G 网络迁移。** 多项研究 [18][19] 展示了 PQC 在 5G 核心网中的集成方案。Alves Faval et al. [18] 在开源 5G 核心（free5GC）中通过 sidecar proxy 模式集成 ML-KEM-768 和 ML-DSA，PQC 将端到端延迟增加至约 54 ms，新增确定性开销约 48-49 ms，方差极小（IQR ≤ 0.2 ms）。

---

## 六、量子计算 × AI/ML

量子机器学习（QML）在理论前景与实际价值之间存在显著张力。

**QKM 综述：审慎乐观。** Tanner et al. [6] 对非变分监督量子核方法进行了全面综述。QKM 使用固定量子特征映射，通过经典凸优化进行模型选择，避免了 barren plateaus。但三大挑战仍然突出：（1）指数浓度（exponential concentration）——量子优势的根本限制；（2）去量子化（dequantization）——经典张量网络方法可模拟某些量子核；（3）核积分算子的谱性质影响泛化能力 [6]。

**非幺正 QML 实证突破。** Masta et al. [7] 通过 570+ 实验横跨四个领域（MNIST 数字分类、PlantVillage 农业病害检测、QM9 分子性质回归、PathMNIST 医学病理），发现非幺正量子层相比幺正基线一致性地提升了 +0.2% 到 +5.8% 的性能。特别值得注意的是医学影像任务中的 **Fisher 效率跃迁**——参数效率在量子比特数从 10 增加到 12 时从负值转为正值，揭示了硬件阈值依赖的效率机制 [7]。

![非幺正量子层 (LCU) 在 MNIST 上不同量子比特规模下三种 CNN 架构的性能对比 [7]](2026-04-15-quantum-frontiers/fig-nonunitary-qml-performance.png)

![Fisher 效率跃迁：PathMNIST 医学影像任务中 12 量子比特处参数效率从负值转为正值 [7]](2026-04-15-quantum-frontiers/fig-fisher-efficiency-transition.png)

![PlantVillage 农业病害分类任务中非幺正量子层的有利扩展特性 [7]](2026-04-15-quantum-frontiers/fig-nonunitary-qml-scaling.png)

**经典模拟的追赶。** Kawase [20] 提出的前向/后向门融合方法使经典 GPU 模拟 QML 速度提升约 20 倍，能够在 20 分钟/epoch 内训练 20 量子比特、1000 层、60,000 参数的模型。这种经典模拟能力的提升对 QML 优势主张形成持续压力。

**量子-经典知识蒸馏。** Yan et al. [21] 展示了大规模 QNN 可通过知识蒸馏压缩为更小架构，减少部署所需的量子比特数和电路深度，同时自蒸馏方法可加速训练收敛——为 QNN 的实际部署提供了新策略。

---

## 七、开放问题与挑战

综合以上文献，量子计算领域面临以下核心开放问题：

1. **纠错开销问题。** 虽然 Google Willow 证明了阈值以下运行的可行性 [1]，但从 105 物理量子比特到数百万物理量子比特（支撑有意义的逻辑计算）之间仍有巨大工程鸿沟。Yang & Murali [22] 的分析表明，当前电路切割技术虽可用于小规模系统，但量子运行时间和经典后处理开销的指数增长使其在较大系统上不可行。

2. **实用量子优势。** 在 random circuit sampling 之外，尚未在具有广泛实际价值的问题上展示无可争辩的量子优势。Alexeev et al. [5] 认为 25-100 逻辑量子比特时代的量子化学可能是第一个突破口。

3. **Barren Plateaus。** 梯度消失仍然是 VQA 和 QML 可扩展性的根本障碍 [3]。虽有缓解策略 [11]，但尚无通用解决方案。

4. **去量子化威胁。** 经典张量网络等方法持续蚕食 QML 声称的优势区域 [6][20]。确定量子优势的精确边界仍是开放问题。

5. **量子网络可扩展性。** 量子中继器尚未实现实用部署 [13]，仿真-现实鸿沟 [4] 限制了量子网络从实验到基础设施的转化。

6. **PQC 迁移紧迫性。** HNDL 攻击模型使 PQC 迁移具有即时紧迫性 [9]。然而，大规模基础设施（特别是 5G/IoT 设备）的迁移面临互操作性和性能挑战 [18][19]。

---

## References

- [1] Google Quantum AI, "Quantum error correction below the surface code threshold", arXiv:2408.13687 (2024); Willow chip announcement, Dec 2024.
- [2] Aasen, Aghaee et al., "Roadmap to fault tolerant quantum computation using topological qubit arrays", arXiv:2502.12252 (2025).
- [3] Wang, Huang et al., "A Review of Variational Quantum Algorithms: Insights into Fault-Tolerant Quantum Computing", arXiv:2604.07909 (2026).
- [4] Gkelias, Burt & Leung, "Quantum Networking Fundamentals: From Physical Protocols to Network Engineering", arXiv:2604.01910 (2026).
- [5] Alexeev, Batista et al., "A Perspective on Quantum Computing Applications in Quantum Chemistry using 25-100 Logical Qubits", arXiv:2506.19337 (2025).
- [6] Tanner, Kam & Wang, "Non-variational supervised quantum kernel methods: a review", arXiv:2604.07896 (2026).
- [7] Masta, Ganguly et al., "Non-Unitary Quantum Machine Learning: Fisher Efficiency Transitions from Distributed Quantum Expressivity", arXiv:2603.27377 (2026).
- [8] Huang, Lin, Luo & Lui, "A Scalable Distributed Quantum Optimization Framework via Factor Graph Paradigm", arXiv:2603.07673 (2026).
- [9] NIST, "NIST Releases First 3 Finalized Post-Quantum Encryption Standards", August 2024. FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA).
- [10] Wikipedia, "Quantum supremacy"; Google Sycamore (2019), Jiuzhang (2020), Zuchongzhi (2021).
- [11] Hirviniemi, Basheer & Cope, "Preventing Barren Plateaus in Continuous Quantum Generative Models", arXiv:2602.10049 (2026).
- [12] Qiao, Li & Liu, "Full-quantum variational dynamics simulation for time-dependent Hamiltonians with global spectral discretization", arXiv:2603.17062 (2026).
- [13] Wikipedia, "Quantum key distribution" and "Quantum network"; satellite QKD 12,900 km (2024), Twin-Field QKD 833.8 km.
- [14] Wikipedia, "Quantum network"; Oxford distributed quantum computing (Feb 2025), Beijing-Shanghai trunk line.
- [15] Swain & Ralph, "Loss-Tolerant Quantum Communication via Bosonic-GKP-Parity-Encoding", arXiv:2604.09002 (2026).
- [16] Chhetri, "Benchmarking NIST-Standardised ML-KEM and ML-DSA on ARM Cortex-M0+", arXiv:2603.19340 (2026).
- [17] Schemitt et al., "Assessing the Impact of Post-Quantum Digital Signature Algorithms on Blockchains", arXiv:2510.09271 (2025).
- [18] Alves Faval, Moreira & Silva, "Empowering Mobile Networks Security Resilience by using Post-Quantum Cryptography", arXiv:2603.28626 (2026).
- [19] Rathi et al., "QORE: Quantum Secure 5G/B5G Core", arXiv:2510.19982 (2025).
- [20] Kawase, "Fast and memory-efficient classical simulation of quantum machine learning via forward and backward gate fusion", arXiv:2603.02804 (2026).
- [21] Yan, Qian, Zhao & Zhang, "Distilling the knowledge with quantum neural networks", arXiv:2603.21586 (2026).
- [22] Yang & Murali, "Understanding the Scalability of Circuit Cutting Techniques for Practical Quantum Applications", arXiv:2411.17756 (2024).
