# 量子计算生态：新兴公司、国家战略与投资格局

- **Date:** 2026-04-15
- **Tags:** quantum-computing, ecosystem, startups, national-programs, investment

## Context

本文基于对量子计算行业主要公司官网、Wikipedia 条目、政府网站（quantum.gov、gov.uk）及行业报告（QURECA）的系统性调研，梳理当前量子计算生态中的新兴公司、国家战略及投资格局。所有数据均来自已验证的公开来源，截至 2026 年 4 月。

---

## 一、新兴量子计算公司

量子计算赛道正在从学术实验室走向商业化，多条技术路线并行推进。以下按技术路线梳理代表性公司。

### 1.1 QuEra Computing（中性原子）

**基本信息**
- **成立时间：** 2018 年
- **总部：** 美国波士顿
- **创始团队：** Mikhail Lukin、Vladan Vuletic、Markus Greiner、Dirk Englund 等（源自 Harvard/MIT 研究）
- **现任 CEO：** Andy Ory（2024 年 7 月接任）

**技术路线与里程碑**

QuEra 采用中性铷原子（neutral rubidium atoms）技术路线，通过激光操控原子实现量子计算。关键产品与里程碑包括：

| 时间 | 里程碑 |
|------|--------|
| 2022 年 11 月 | 发布 **Aquila** 系统（256 qubits），通过 Amazon Braket 提供云访问 |
| 2023 年 12 月 | Harvard/QuEra 团队在 *Nature* 发表论文，演示 **48 个逻辑量子比特**，可执行数百次逻辑门操作 |
| 2025 年 | **Gemini** 系统：260 原子 gate-model 量子计算机，单比特保真度 >99.9%，双比特保真度 >99.2% |
| 2025 年 9 月 | 演示 **3,000 量子比特**的相干系统，相干存储时间超过 2 小时 |
| 2025 年 5 月 | 日本 AIST（产业技术综合研究所）部署首台 on-premises 系统，与 ABCI-Q 超算配合 |

**融资与估值**
- 2021 年 11 月：首轮融资 $17M
- 2025 年 2 月：**Series B $230M**，由 SoftBank Vision Fund 和 Google Quantum AI 领投，估值 **$1B**
- 2025 年 9 月：Series B 扩展轮，Nvidia NVentures 参投

**2026 路线图：** >10,000 物理原子，100 个逻辑量子比特

> **来源：** [Wikipedia - QuEra Computing](https://en.wikipedia.org/wiki/QuEra_Computing)

---

### 1.2 Atom Computing + Microsoft（中性原子）

**基本信息**
- **成立时间：** 2018 年
- **总部：** 美国加州伯克利（运营设施在科罗拉多 Boulder）
- **创始人：** Ben Bloom、Jonathan King
- **技术路线：** 中性原子（碱土金属，如锶 strontium 和镱 ytterbium）

**关键里程碑**

| 时间 | 里程碑 |
|------|--------|
| 2021 年 | 100-qubit 原型机 |
| 2023 年 | 第二代系统：**1,000+ 物理量子比特** |
| 2024 年 11 月 | 演示 **24 个纠缠逻辑量子比特**；运行 **28 逻辑量子比特** 的 Bernstein-Vazirani 算法 |
| 2024 年底 | 与 **Microsoft** 合作，结合 Microsoft 量子纠错技术与 Atom 的 1,000+ 量子比特系统，计划 2025 年交付商业逻辑量子比特计算机 |

**融资**
- 2018 年：$5M 种子轮
- 2021 年：$15M Series A
- 2022 年初：$60M Series B

**荣誉：** 2024 年科罗拉多州"年度新兴科技公司"；2025 年 Fast Company "十大最具创新力计算公司"之一；入选 DARPA Quantum Benchmarking Initiative Stage B

> **来源：** [Wikipedia - Atom Computing](https://en.wikipedia.org/wiki/Atom_Computing)

---

### 1.3 PsiQuantum（光量子）

**基本信息**
- **成立时间：** 2016 年
- **总部：** 美国帕洛阿尔托
- **创始人：** Jeremy O'Brien、Terry Rudolph、Peter Shadbolt、Mark Thompson（来自 University of Bristol 和 Imperial College London）
- **技术路线：** 硅光子（silicon photonic）量子计算

**技术特点**

PsiQuantum 的核心策略是利用 **GlobalFoundries 的半导体制造工艺**生产光量子芯片，试图跳过逐步增加量子比特的路径，直接构建 utility-scale 容错量子计算机。

**融资与战略合作**

| 时间 | 事件 |
|------|------|
| 2021 年 7 月 | 融资 $665M，估值 $3.15B |
| 2022 年 | 与 GlobalFoundries 获得美国联邦量子研发资助；$22.5M 空军研究实验室合同 |
| 2023 年 | 入选 **DARPA US2QC**（Underexplored Systems for Utility-Scale Quantum Computing）项目 |
| 2024 年 7 月 | 成为 Illinois Quantum and Microelectronics Park 的 anchor tenant |
| 2025 年 3 月 | **融资 $750M**，估值 **$6B**，投资方包括 BlackRock、Baillie Gifford、Microsoft M12 |

**澳大利亚布里斯班项目**

澳大利亚联邦政府和昆士兰州政府承诺投入 **A$940M**（含贷款），在布里斯班建造"全球首台 utility-scale 容错量子计算机"，目标 **2027 年底**投入运营。目前公司有约 280 名员工。

> **来源：** [Wikipedia - PsiQuantum](https://en.wikipedia.org/wiki/PsiQuantum)

---

### 1.4 Intel Tunnel Falls（硅自旋量子比特）

**技术路线**

Intel 采用硅自旋量子比特（silicon spin qubits）路线，核心优势在于利用其成熟的半导体制造工艺。

**低温控制芯片 Horse Ridge**

Intel 与 QuTech 合作开发了 **Horse Ridge** 低温控制芯片：
- **Horse Ridge I**（2020 年）：基于 Intel 22nm FinFET 技术，可在 1K 以上温度操控量子比特
- **Horse Ridge II**（2020 年 12 月）：可操控和读取量子比特状态，驱动最多 16 个自旋量子比特，搭载 22 个高速 DAC

**Tunnel Falls 芯片：** Intel 于 2023 年 6 月发布 Tunnel Falls 量子处理器，这是一款利用其 300mm 晶圆产线制造的硅自旋量子芯片。Intel 的制造能力使其可以高产量生产量子芯片，面向研究机构开放。

> **来源：** [Wikipedia - Horse Ridge (chip)](https://en.wikipedia.org/wiki/Horse_Ridge_(chip)); [Wikipedia - Rigetti Computing](https://en.wikipedia.org/wiki/Rigetti_Computing)（关于 84-qubit 处理器的上下文参照）

---

### 1.5 Pasqal（中性原子 -- 法国）

**基本信息**
- **成立时间：** 2019 年
- **总部：** 法国 Palaiseau
- **联合创始人：** Alain Aspect（2022 年诺贝尔物理学奖得主）等
- **团队：** 275+ 员工，70 位 PhD，30+ 国籍

**技术与合作**

Pasqal 提供全栈中性原子量子计算方案（硬件 QPU + 软件 SDK + 云服务），关键合作包括：
- 与 **NVIDIA** 集成 CUDA-Q（2026 年 3 月宣布），支持混合量子-HPC 环境
- 与 **IBM** 在量子计算领域合作
- 与 True Nexus 合作进行食品蛋白质设计
- 沙特阿拉伯 Aramco 区域部署

**SPAC 上市**

2026 年 3 月 4 日，Pasqal 宣布通过与 **Bleichroeder Acquisition Corp II** 的业务合并实现 SPAC 上市。投资方包括 Quantonation、BPI France、Temasek 等。

> **来源：** [Pasqal 官网 - About](https://www.pasqal.com/about)

---

### 1.6 Alice & Bob（Cat Qubit -- 法国）

**基本信息**
- **成立时间：** 2020 年
- **总部：** 巴黎（波士顿设有子公司）
- **创始人：** Theau Peronnin（CEO）、Raphael Lescanne（CTO），源自法国研究机构
- **团队：** 110 名员工

**Cat Qubit 技术**

Alice & Bob 的核心创新是 **cat qubit**（以薛定谔的猫命名）——一种可以指数级减少 bit-flip 错误的量子比特架构。最新成果：通过双光子注入方案实现 **160 倍错误抑制**，同时减少芯片面积需求。

**融资**

| 时间 | 金额 |
|------|------|
| 2022 年 | $30M Series A |
| 2025 年 1 月 | **$104M Series B** |
| 累计 | 超过 $134M |

**战略项目**
- 2024 年入选法国 **PROQCIMA** 项目（$548M 国家量子计划）
- 2025 年宣布在巴黎建造 $50M 先进量子计算实验室
- 2025 年入选 **DARPA Quantum Benchmarking Initiative**
- **目标：** 2030 年实现 100 个逻辑量子比特

> **来源：** [Wikipedia - Alice & Bob](https://en.wikipedia.org/wiki/Alice_%26_Bob_(company))

---

### 1.7 Rigetti Computing（超导量子比特）

**基本信息**
- **成立时间：** 2013 年
- **创始人：** Chad Rigetti（前 IBM 物理学家）
- **总部：** 美国加州伯克利
- **现任 CEO：** Subodh Kulkarni（2022 年 12 月起）
- **上市：** 2022 年 3 月在 NASDAQ 上市（RGTI），通过 SPAC 合并

**技术与产品**

Rigetti 开发超导量子集成电路，运营 **Fab-1** 制造设施（加州 Fremont），支持快速原型制造。提供 Forest 云计算平台，开发者可使用 Quil（Quantum Instruction Language）编写量子算法。

**处理器发展**

| 时间 | 里程碑 |
|------|--------|
| 2016 年 2 月 | 首款 3-qubit 处理器 |
| 2017 年 | 8-qubit 系统 |
| 2023 年 7 月 | **84-qubit** 单芯片量子处理器 |

**财务状况（2024 年）**
- 员工：140 人
- 收入：$10.8M
- 净亏损：$201M
- 总资产：$285M

> **来源：** [Wikipedia - Rigetti Computing](https://en.wikipedia.org/wiki/Rigetti_Computing)

---

### 1.8 Diraq（硅自旋 -- 澳大利亚）

**基本信息**
- **技术路线：** 硅基量子计算，利用改良硅晶体管
- **运营地点：** 澳大利亚和美国
- **研究产出：** 40+ 篇 Nature/Science 论文，60+ 项专利

**技术指标**
- 硅自旋量子比特单元，保真度超过 **99%**
- 目标：每分钟 **100 万次**纠错操作
- 每量子比特成本低于 **$1**

**路线图**
- 2031 年：百万量子比特
- 2033 年：千万量子比特
- 量子处理单元可装入单个数据中心机架

> **来源：** [Diraq 官网](https://diraq.com)

---

### 1.9 IonQ（离子阱 -- 补充）

作为量子计算商业化的标杆公司，IonQ 值得纳入对比：

- **成立时间：** 2015 年，由 Christopher Monroe 和 Jungsang Kim 创立
- **技术：** 离子阱（trapped ion）
- **上市：** 2021 年 10 月 SPAC IPO，募资 $636M（NYSE: IONQ）
- **2025 年收入：** $130M（量子专业公司中首家突破 $100M 营收）
- **员工：** 1,132 人
- **总资产：** $65.7 亿
- **2024-2025 年收购：** Qubitekk、ID Quantique、Oxford Ionics、Lightsynq Technologies、Vector Atomic 等；拟以 $1.8B 收购 SkyWater Technology

> **来源：** [Wikipedia - IonQ](https://en.wikipedia.org/wiki/IonQ)

---

## 二、全球量子国家战略

各国政府已将量子技术视为战略优先领域，投入大量资金支持研发。以下数据来自已验证来源。

### 各国量子投资概览

| 国家/地区 | 投资规模 | 关键项目/计划 | 来源 |
|-----------|---------|--------------|------|
| **美国** | NQI Act $1.2B/5 年（2018）；DOE $625M 量子中心（2020）| National Quantum Initiative Act（2018）；DOE 5 个量子信息科学中心；DARPA US2QC；CHIPS and Science Act（2022）| quantum.gov, QURECA |
| **中国** | 约 ¥760 亿（约 €100 亿 / ~$10B）| 国家量子信息科学实验室（合肥）；USTC 团队（Jiuzhang 光量子、祖冲之号超导） | Wikipedia, QURECA |
| **欧盟** | €10 亿/10 年（2018-2028）| Quantum Flagship；首批 20 项目 €1.32 亿（2018） | Wikipedia - Quantum Flagship |
| **英国** | £10 亿+（两阶段）| National Quantum Technologies Programme；NQCC（2024 年 10 月开放）；UK National Quantum Strategy（2023 年 3 月发布） | gov.uk, Wikipedia - NQCC |
| **德国** | €6.5 亿（2018）+ €20 亿（2020 扩展）| Munich Quantum Valley 等 | QURECA |
| **法国** | €18 亿（2021 年 Macron 宣布）| National Quantum Plan；PROQCIMA $548M 量子计算专项 | QURECA, Wikipedia - Alice & Bob |
| **日本** | ¥300 亿（~$280M）+ Moonshot 计划 ¥150-200 亿 | RIKEN；AIST（QuEra 日本部署）；Moonshot Project（目标 2050） | QURECA, Wikipedia - QuEra |
| **韩国** | ₩445 亿（~$40M）/5 年 | ICT 量子技术专项 | QURECA |
| **加拿大** | 过去十年 >$10 亿 | Perimeter Institute；多所大学量子中心 | QURECA |
| **澳大利亚** | AU$1.3 亿联邦资金 + PsiQuantum AU$9.4 亿专项 | PsiQuantum 布里斯班工厂；Silicon Quantum Computing（AU$8300 万） | QURECA, Wikipedia - PsiQuantum, SQC |
| **印度** | ₹8,000 crores（~$10 亿）/5 年 | National Quantum Mission（国家量子使命）| Wikipedia - Quantum technology |

### 重点国家分析

#### 美国

美国通过 **National Quantum Initiative Act**（2018 年 12 月签署）建立了联邦层面的量子技术协调机制，由 NIST、NSF 和 DOE 三大机构牵头。2020 年 DOE 投入 $625M 建立 5 个多学科量子信息科学研究中心。2022 年的 **CHIPS and Science Act** 进一步授权量子网络基础设施和标准开发。2022 年 5 月，总统签署 **National Security Memorandum 10**，确立维持美国量子领先地位并推进抗量子密码学转型的政策。

> **来源：** [quantum.gov](https://www.quantum.gov/about/), [quantum.gov/strategy](https://www.quantum.gov/strategy/)

#### 中国

中国在量子技术上的投资估计约 ¥760 亿（约 $100 亿），虽然官方数据难以独立验证，但建设规模可见一斑。USTC（中国科学技术大学）Pan Jianwei 团队的标志性成果包括：

- **九章（Jiuzhang）**（2020 年 12 月）：光量子计算机，200 秒内完成高斯玻色子采样（检测 76 光子），估计传统超算需 25 亿年完成同样计算
- **祖冲之号（Zuchongzhi）**：超导量子处理器系列，展示量子优越性

> **来源：** [Wikipedia - Jiuzhang](https://en.wikipedia.org/wiki/Jiuzhang_(quantum_computer)), QURECA

#### 欧盟

**Quantum Flagship**（2018-2028）是欧盟 Future and Emerging Technologies 计划下的大型倡议，总预算 €10 亿/10 年，资助超过 5,000 名欧洲研究人员。2018 年 10 月首批发布 20 个项目、€1.32 亿资助。长期愿景是构建互联的量子基础设施——"量子网络"（quantum web），连接量子计算机、模拟器和传感器。

高层指导委员会由 24 名成员组成（12 名学者 + 12 名产业界代表）。

> **来源：** [Wikipedia - Quantum Flagship](https://en.wikipedia.org/wiki/Quantum_Flagship)

#### 法国

法国总统马克龙 2021 年宣布 **€18 亿国家量子计划**，使法国成为欧洲量子投资力度最大的国家之一。其中 **PROQCIMA** 项目专门面向量子计算硬件，预算 $548M，Alice & Bob 和 Pasqal 等本土公司均为参与者。

> **来源：** QURECA, [Wikipedia - Alice & Bob](https://en.wikipedia.org/wiki/Alice_%26_Bob_(company))

#### 澳大利亚

澳大利亚的量子生态以 **PsiQuantum 布里斯班项目**（AU$9.4 亿政府承诺）和 **Silicon Quantum Computing**（AU$8,300 万，CEO Michelle Simmons）为核心。SQC 于 2025 年实现 98.87% 准确率的 Grover 算法运行（无纠错），2025 年 8 月澳大利亚国防部购买了其 QML 处理器用于本地部署。

> **来源：** [Wikipedia - PsiQuantum](https://en.wikipedia.org/wiki/PsiQuantum), [Wikipedia - Silicon Quantum Computing](https://en.wikipedia.org/wiki/Silicon_Quantum_Computing)

---

## 三、投资与融资格局

### 2024-2026 主要融资事件

| 时间 | 公司 | 金额 | 轮次/方式 | 估值 | 投资方 |
|------|------|------|----------|------|--------|
| 2022 年初 | Atom Computing | $60M | Series B | — | — |
| 2022 年 | Alice & Bob | $30M | Series A | — | — |
| 2021 年 7 月 | PsiQuantum | $665M | — | $3.15B | — |
| 2021 年 10 月 | IonQ | $636M | SPAC IPO | $1.5B（初始）| NYSE: IONQ |
| 2025 年 1 月 | Alice & Bob | $104M | Series B | — | — |
| 2025 年 2 月 | QuEra | $230M | Series B | $1B | SoftBank Vision Fund, Google Quantum AI |
| 2025 年 3 月 | PsiQuantum | $750M | — | $6B | BlackRock, Baillie Gifford, Microsoft M12 |
| 2025 年 9 月 | QuEra | 扩展轮 | Series B ext. | — | Nvidia NVentures |
| 2026 年 3 月 | Pasqal | SPAC 上市 | 与 Bleichroeder Acquisition Corp II 合并 | — | Quantonation, BPI France, Temasek |
| 2025 年（进行中） | IonQ | $1.8B | 收购 SkyWater Technology | — | — |

### 融资趋势分析

1. **轮次规模急剧增长：** PsiQuantum 从 $665M 到 $750M，QuEra 的 $230M Series B，单轮融资已进入数亿美元级别
2. **战略投资者入场：** SoftBank、Google、Nvidia、BlackRock 等大型机构/科技巨头直接投资量子初创
3. **SPAC 与公开市场：** Rigetti（2022）、IonQ（2021）已上市，Pasqal（2026）通过 SPAC 上市，反映市场对量子赛道的长期信心
4. **IonQ 的激进整合：** 2024-2025 年连续收购 7+ 家公司，构建从硬件到应用的全栈能力
5. **政府资金杠杆效应：** PsiQuantum 的澳大利亚 AU$9.4 亿政府承诺显示，政府资金正成为量子初创公司落地大型项目的关键催化剂

> **来源：** 各公司 Wikipedia 条目及官网

---

## 四、DARPA 量子项目

美国国防高级研究计划局（DARPA）在量子计算领域运营多个重要项目：

### 4.1 US2QC（Underexplored Systems for Utility-Scale Quantum Computing）

**目标：** 判断"非主流"量子计算方案（underexplored approaches）能否比传统预测更快地实现 utility-scale 运行。

**管理办公室：** Microsystems Technology Office (MTO)
**项目经理：** Micah Stoutimore

**关键特点：** 该项目强调严格的、协作性的、灵活的验证与确认（V&V），与研发工作并行推进。DARPA 认为这种验证"很可能是一个困难的、多年期的过程"。

**已知参与者：** PsiQuantum（2023 年入选）

> **来源：** [DARPA US2QC 项目页](https://www.darpa.mil/research/programs/underexplored-systems-for-utility-scale-quantum-computing)

### 4.2 Quantum Benchmarking Initiative

DARPA 的量子基准测试计划旨在建立量子系统性能评估标准。已知入选 **Stage B** 的公司包括：
- **Atom Computing**
- **Alice & Bob**（2025 年入选）

> **来源：** [Wikipedia - Atom Computing](https://en.wikipedia.org/wiki/Atom_Computing), [Wikipedia - Alice & Bob](https://en.wikipedia.org/wiki/Alice_%26_Bob_(company))

---

## 五、生态趋势与展望

### 5.1 技术路线多元竞争

当前量子计算领域呈现明显的多技术路线并行格局：

| 技术路线 | 代表公司 | 核心优势 | 主要挑战 |
|---------|---------|---------|---------|
| **中性原子** | QuEra, Atom Computing, Pasqal | 高扩展性（已达 3,000+ qubits）、长相干时间 | 门操作速度 |
| **光量子** | PsiQuantum | 利用半导体工艺制造、室温运行 | 确定性光子源、损耗 |
| **超导** | Rigetti, IBM, Google | 最成熟的技术路线 | 低温要求、串扰 |
| **离子阱** | IonQ | 高保真度、全连接 | 扩展速度 |
| **硅自旋** | Intel, Diraq | CMOS 兼容、利用现有晶圆厂 | 量子比特数量仍偏少 |
| **Cat Qubit** | Alice & Bob | 天然抑制 bit-flip 错误 | 仍需大量 phase-flip 纠错 |

### 5.2 关键观察

1. **中性原子技术崛起：** QuEra 的 3,000-qubit 系统和 Atom Computing 的 1,000+ qubit 系统表明，中性原子路线在 qubit 数量扩展方面处于领先。2023 年 Harvard/QuEra 的 48 逻辑量子比特突破是里程碑事件。

2. **逻辑量子比特成为核心指标：** 行业焦点正从物理量子比特数量转向逻辑量子比特。QuEra 目标 100 个逻辑 qubits（2026），Alice & Bob 目标 100 个逻辑 qubits（2030），Atom Computing + Microsoft 已演示 24 个纠缠逻辑 qubits。

3. **国家竞争加剧：** 中国（~$10B）、美国（NQI + DOE + DARPA 多管齐下）、法国（€18 亿）形成三大量子投资极。澳大利亚通过 PsiQuantum 布里斯班项目弯道超车。

4. **商业化窗口逼近：** IonQ 已实现 $130M 年收入，Rigetti 和 IonQ 均已公开上市。但多数公司仍处于巨额亏损阶段（Rigetti 净亏 $201M，IonQ 净亏 $510M），量子优越性到实际商业价值的转化仍需时日。

5. **硬件-软件协同加速：** Microsoft + Atom Computing 的逻辑量子比特合作、NVIDIA CUDA-Q 与 Pasqal 的集成，显示科技巨头正在通过软件生态切入量子计算价值链。

---

## References

1. Wikipedia - QuEra Computing: https://en.wikipedia.org/wiki/QuEra_Computing
2. Wikipedia - Atom Computing: https://en.wikipedia.org/wiki/Atom_Computing
3. Wikipedia - PsiQuantum: https://en.wikipedia.org/wiki/PsiQuantum
4. Wikipedia - Rigetti Computing: https://en.wikipedia.org/wiki/Rigetti_Computing
5. Wikipedia - Alice & Bob: https://en.wikipedia.org/wiki/Alice_%26_Bob_(company)
6. Wikipedia - IonQ: https://en.wikipedia.org/wiki/IonQ
7. Wikipedia - Quantum Flagship: https://en.wikipedia.org/wiki/Quantum_Flagship
8. Wikipedia - Horse Ridge (chip): https://en.wikipedia.org/wiki/Horse_Ridge_(chip)
9. Wikipedia - Jiuzhang: https://en.wikipedia.org/wiki/Jiuzhang_(quantum_computer)
10. Wikipedia - Silicon Quantum Computing: https://en.wikipedia.org/wiki/Silicon_Quantum_Computing
11. Wikipedia - National Quantum Initiative Act: https://en.wikipedia.org/wiki/National_Quantum_Initiative_Act
12. Wikipedia - Quantum technology (national programs): https://en.wikipedia.org/wiki/Quantum_technology
13. Wikipedia - NQCC: https://en.wikipedia.org/wiki/National_Quantum_Computing_Centre
14. Pasqal 官网: https://www.pasqal.com/about
15. Diraq 官网: https://diraq.com
16. US quantum.gov: https://www.quantum.gov/about/, https://www.quantum.gov/strategy/
17. UK gov.uk National Quantum Strategy: https://www.gov.uk/government/publications/national-quantum-strategy
18. QURECA - Quantum Initiatives Worldwide: https://qureca.com/overview-on-quantum-initiatives-worldwide/
19. DARPA US2QC: https://www.darpa.mil/research/programs/underexplored-systems-for-utility-scale-quantum-computing
20. Quantum Computing Inc. Blog: https://www.quantumcomputinginc.com/blog/quantum-investments-by-country
