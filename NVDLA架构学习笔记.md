# NVDLA（NVIDIA NPU）架构学习笔记

> 学习目的：为搭建 NPU FPGA 原型验证系统提供参考架构
> 创建日期：2026-08-12
> 内容来源：NVDLA 官方文档（Primer / Unit Description）+ 公开资料整理

---

## 0. 澄清："NV 的 NPU"指什么

NVIDIA 官方没有直接叫"NPU"的产品，业界所指的 NV NPU 有四条不同的技术路线：

| 名称 | 出处 | 架构路线 | 参考价值 |
|---|---|---|---|
| **NVDLA** | 2017 年开源，基于 Xavier SoC DLA（Git：[github.com/nvdla/hw](https://github.com/nvdla/hw)） | 专用 CNN 推断加速器，**RTL 全开源** | ⭐ 唯一可直接研究/移植 |
| **DLA 2.0** | Jetson Orin / Thor 内部 | NVDLA 的闭源演进版 | 只有公开规格，无 RTL |
| **Tensor Core** | GPU 内（Volta → Blackwell） | SIMT GPU 内的张量单元 | 与"NPU"是不同物种 |
| **GB10 NPU** | DGX Spark（2025） | 闭源 | 无参考价值 |

**结论：FPGA 原型验证场景下，研究 NVDLA 即可。** 它是唯一开源、唯一有官方 FPGA 移植（AWS F1 / Xilinx 端口、FireSim 集成）的 NV 推断加速器，是"NPU 的教科书级架构"。

⚠️ **仓库状态**：`nvdla/hw` 的 `nvdlav1` 分支为稳定维护分支（固定 2048 个 8-bit MAC），最后一次推送 **2022-03**，开发已实质停止（NVIDIA 转向闭源 DLA 2.0）。作为参考架构价值不减，且设计冻结不会有大改动。

---

## 1. 总体架构

```
                     ┌───────────────── NVDLA Core ─────────────────┐
   SoC/CPU ─── CSB ─▶│ 配置空间 (Configuration Space)                 │
 (配置总线,            │   每个子模块有独立寄存器组, 乒乓寄存器          │
  类似 AHB-Lite)      │                                              │
                     │  ┌────── Convolution Pipeline (5级) ───────┐  │
                     │  │  CDMA → CBUF → CSC → CMAC → CACC        │  │
                     │  │        512KB    调度    MAC   累加器     │  │
                     │  └─────────────────────────────────────────┘  │
                     │              │ fused 直连(不落 DRAM)           │
                     │  ┌──── Partition P (后处理/搬运) ──────────┐  │
                     │  │  SDP → PDP → CDP   共享 CMEM 512KB      │  │
                     │  │  RUBIK(数据重整)   BDMA(批量搬运)         │  │
                     │  └─────────────────────────────────────────┘  │
   DRAM ─── AXI(DBB)─▶│  Memory Controller Interface (MCIF)         │
   (数据总线, AXI4)    └──────────────────────────────────────────────┘
```

**核心设计思想**：把计算流水线（卷积 + 后处理）和存储搬运完全解耦。每个单元既可以独立工作（memory-to-memory），也可以流水线融合（on-the-fly），由 CPU 通过寄存器逐层配置每个 layer 的行为 —— **per-layer programming 模型**。

这与 GPU 的通用 SIMT 完全不同：NVDLA 是为推断场景定制的专用机（无训练支持、无 scatter/gather、面向小型网络、每层编程）。

---

## 2. 卷积流水线（核心中的核心）

五级流水，每级一个独立 CSB 从端口：

| 级 | 模块 | 职责 | 关键设计点 |
|---|---|---|---|
| 1 | **CDMA** | 从 DRAM 取输入激活、权重、bias | 卷积专用 DMA，与 BDMA 不同 |
| 2 | **CBUF** | 512KB SRAM（32 bank，每 bank 两块 SRAM） | 权重和激活**分 bank 存储**避免冲突；缓存不下的激活要重复读 DRAM N 次 |
| 3 | **CSC** | 卷积调度器 | 从 CBUF 按调度序喂数据，处理 padding / zero-fill |
| 4 | **CMAC** | MAC 阵列 | 见下 |
| 5 | **CACC** | 累加器 | 收集部分和（partial sum），进位处理，装配后交给 SDP |

### 2.1 CMAC 组织方式（理解所有 AI 芯片的钥匙）

- 16 个 MAC cell × 每 cell 128 个 INT8 MAC = **2048 MAC**
- 每 cell 实际是 64 个 16-bit 乘法器 + 72 个加法器（Winograd 后处理用）
- 16-bit 乘法器可拆成两个 8-bit 使用 → **INT8 吞吐是 INT16/FP16 的两倍**（2048 vs 1024 MAC）
- 支持 **Winograd 变换**（5×5 及以下卷积核可大幅减少乘法次数）
- MAC cell 可独立门控时钟：kernel 数不足时自动关掉空闲 cell（省功耗）
- 物理上拆成 CMAC_A / CMAC_B 两个相同半区，各有独立 CSB 接口

### 2.2 三级数据复用（性能的关键，也是自研 NPU 必抄的作业）

1. **权重复用**：权重驻留在 CMAC 寄存器，跨多个激活复用
2. **激活复用**：同一像素广播给多个输出通道的 MAC
3. **累加器复用**：部分和留在 CACC 累加，减少写回 DRAM

### 2.3 Full 配置实测（16nm @ 1GHz）

| 指标 | 数值 |
|---|---|
| INT8 MAC 数 | 2048（INT16/FP16 为 1024） |
| CBUF | 512KB |
| 面积 | ~3.3 mm² |
| 外部带宽 | 20 GB/s（够跑 ResNet-50 269fps） |
| 功耗 | 388 mW |
| 能效 | 5.4 DL TOPS/W |

**注意**：这是带宽受限设计 —— 20GB/s 带宽就能跑 ResNet-50 269fps，说明数据复用做得好。

---

## 3. 后处理与数据搬运（Partition P）

| 模块 | 全称 | 功能 | 要点 |
|---|---|---|---|
| **SDP** | Single Data Processor | 逐元素运算 | 内部 X1/X2/C 三子单元串行：BN、bias、scale、ReLU/PReLU、截断。**卷积结果必须经 SDP 才能写回 DRAM**（CACC 无直出 DMA），可 fused 在卷积后 on-the-fly 处理 |
| **PDP** | Planar Data Processor | 池化 | max / min / mean，逐平面操作；可从 DRAM 独立读取，也可直接吃 SDP 输出（fused 模式） |
| **CDP** | Channel Data Processor | 跨通道运算 | 基于 **LUT 引擎**实现 LRN（AlexNet/GoogLeNet 需要）；LUT 也是 sigmoid/tanh 的实现手段 |
| **RUBIK** | 数据重整器 | 布局变换 | reshape / transpose / NCHW↔NHWC 等，支持 INT8/16/FP16，带 DMA |
| **BDMA** | Bridge DMA | 批量搬运 | 片上 SRAM ↔ DRAM，**旁路卷积流水线**，可与其他单元并发执行 |

**CMEM 512KB**：SDP/PDP/CDP 共用的工作存储（后处理分区的临时表面缓存）。

**与 CBUF 的分工**：CBUF 管卷积计算的数据流，CMEM 管后处理的数据流。

---

## 4. 总线与编程模型

### 4.1 CSB（Configuration Space Bus）—— 配置总线

- 低带宽简单同步总线：req / wr / addr / wdata / wmask + rsp / rdata 握手
- 与 APB / AHB-Lite 同级别（复用已有总线知识）
- 每个子模块一套寄存器组
- **乒乓（ping-pong）寄存器**支持流水线重叠：上一 layer 运行时 CPU 即可配置下一 layer

### 4.2 DBB（Data Base Bus）—— AXI 数据总线

- AXI4 高带宽接口，支持 outstanding 事务
- 与 pulp-platform/axi 学习内容直接通用（高性能 master：burst、outstanding、QoS）

### 4.3 两种运行模式（贯穿所有单元）

| 模式 | 行为 | 用途 |
|---|---|---|
| **独立模式** | 单元从 DRAM 读、写回 DRAM（memory-to-memory） | 单算子验证 |
| **融合模式** | 单元输出直接喂给下一单元（on-the-fly） | 省 DRAM 带宽：SDP 融合卷积、PDP 融合 SDP |

### 4.4 软件栈（验证 NPU 与验证普通 IP 的最大区别）

- **Compiler**（TVM 系）：模型 → loadable 二进制（含每个 layer 的寄存器配置 + 数据调度）
- **UMD**（用户态驱动）：加载网络、提交任务
- **KMD**（内核态驱动）：Linux DRM / GEM PRIME 管理 DMA buffer
- KMD 将 BDMA、CONV、SDP、PDP、CDP、RUBIK 视为 **6 个独立 processor** 做任务管理

> 推论：没有软件栈，RTL 就算正确也无法端到端验证 —— 这是 NPU 原型验证和普通 IP 验证的本质区别。

---

## 5. 关键参数汇总（Full 配置）

| 项 | 值 |
|---|---|
| MAC 阵列 | 2048 INT8 / 1024 INT16·FP16 |
| 卷积核 | ≤ 7×7（Winograd 优化 5×5 以下） |
| CBUF | 512KB（32 bank × 16KB，每 bank 两块 SRAM） |
| CMEM | 512KB |
| 数据类型 | INT8 / INT16 / FP16 |
| 累加精度 | INT32 / FP32 |
| 池化 | max / min / mean，窗口步长可配置（上限 8×8） |
| 激活 | ReLU / PReLU、BN、bias、scale、LUT（sigmoid / tanh / LRN） |
| 总线 | CSB（配置）+ DBB AXI4（数据） |
| 可配置范围 | MAC 8 ~ 4096、buffer 128KB ~ 512KB（SMALL/LARGE/FULL 三档） |
| 电源域 | pd_abuf / pd_cbuf / pd_cmac / pd_sdp / pd_pdp / pd_cdp / pd_rbk / pd_bdma / pd_glb / pd_axi（10 个，ASIC 向） |

---

## 6. 对 FPGA 原型验证系统的启示

### 6.1 资源估算（硬件选型第一件事）

- 2048 INT8 MAC ≈ **1024 个 DSP48**（每个可做 2 个 INT8 MAC），或 LUT 实现
- 1MB SRAM（CBUF + CMEM）≈ **8000+ 块 BRAM（36Kb）**
- 结论：Full 配置中小型 FPGA 放不下 → **先降配置（512 MAC 级）或 LUT 化 MAC**

### 6.2 可配置性是设计出来的一部分

NVDLA 从 8 到 4096 MAC 可缩放（小配置 64 MAC / 128KB；大配置 1024 INT8 / 256KB；全配置 2048 / 512KB）。这种"参数化配置 → 生成实例"的架构方法值得自研时借鉴：**先在 FPGA 验证小配置，再推大配置**。

### 6.3 验证方法学（官方环境可整套抄走）

| 层 | 官方资源 | 说明 |
|---|---|---|
| 黄金参照 | C model（`nvdla/hw` 自带） | RTL + C model + testbench 一体 |
| 验证环境 | `hw/verif/` UVM testbench | 完整 UVM 环境 |
| 软硬件协同 | compiler + KMD/UMD | 端到端链路 |
| 性能预估 | 官方配置级性能估算器 | 验证前先估算带宽够不够 |
| FPGA 现成环境 | `nvdla/vp_awsfpga`、`nvdla/firesim-nvdla` | AWS FPGA 虚拟平台 / RISC-V Rocket Chip + FireSim |

### 6.4 FPGA 移植的典型坑

- 时钟门控 / 电源域结构是为 ASIC 设计的 → 移植时砍掉
- Winograd 加法阵列、LUT 引擎在 FPGA 上资源开销大 → 可先禁用

### 6.5 建议路线

1. 以 NVDLA 框架（五级卷积流水线 + 后处理分区 + CSB/AXI 双总线 + 乒乓配置）为自研 NPU 参考模板
2. 验证系统照抄三层结构：**C model 黄金参照 + UVM + 软硬件协同**
3. 先跑通小配置，再评估 Full 配置的 FPGA 资源预算

---

## 7. 学习资源清单

### 官方仓库（github.com/nvdla 组织）

| 仓库 | 内容 | 状态 |
|---|---|---|
| [nvdla/hw](https://github.com/nvdla/hw) | RTL + C model + testbench + 综合脚本 + 性能估算器 | 2022-03 后停止 |
| [nvdla/sw](https://github.com/nvdla/sw) | compiler + runtime + KMD/UMD 驱动 | 2019-09 后停止 |
| [nvdla/vp_awsfpga](https://github.com/nvdla/vp_awsfpga) | AWS FPGA 虚拟平台 | — |
| [nvdla/firesim-nvdla](https://github.com/nvdla/firesim-nvdla) | FireSim（RISC-V Rocket Chip）集成 | — |

### 文档

- [NVDLA Primer（官方架构白皮书）](https://nvdla.org/primer.html)
- [Unit Description（单元详细说明）](https://nvdla.org/hw/v1/ias/unit_description.html)
- [NVDLA 核心架构（DeepWiki）](https://deepwiki.com/nvdla/hw/2.1-core-architecture)
- [NVDLA 处理单元（DeepWiki）](https://deepwiki.com/nvdla/hw/2.3-processing-units)
- [NVIDIA DLA HotChips 30 演讲 PDF](https://old.hotchips.org/hc30/2conf/2.08_NVidia_DLA_Nvidia_DLA_HotChips_10Aug18.pdf)

### 中文资料

- [NVDLA专题：Convolution Pipeline 模块介绍（CSDN）](https://blog.csdn.net/fangfanglovezhou/article/details/141167033)
- [NVDLA 软件架构解析——内核驱动（CSDN）](https://blog.csdn.net/devcloud/article/details/91523479)
- [NVDLA 硬件结构（知乎专栏）](https://zhuanlan.zhihu.com/p/404723321)

### 本地资料

- [NVDLA架构学习笔记.md](NVDLA架构学习笔记.md) — 本笔记
- [NVDLA架构图解.html](NVDLA架构图解.html) — 可视化架构图（SoC 集成视图 / 内部模块 / 卷积数据流 / CMAC 阵列，浏览器打开，自动适配明暗主题）

---

## 8. 补充：常见缩写澄清 —— TS 与 SM

> 起因：学习过程中遇到"**TS 访存 / SM 访存**"的说法。经查证，**TS 和 SM 都不是 NVDLA 的官方标准术语**（官方缩写表只有 CSB/DBB/CBUF/CMEM/CDMA/CSC/CMAC/CACC/SDP/PDP/CDP/RUBIK/BDMA），含义完全取决于资料来源语境，在此记录备查。

### 8.1 SM

| 语境 | 含义 | 把握度 |
| --- | --- | --- |
| GPU 架构资料（主流） | **Streaming Multiprocessor（流多处理器）**：NVIDIA GPU 的基本计算单元，内含 CUDA 核心、Tensor Core、Warp Scheduler、寄存器堆、Shared Memory/L1 | ⭐ 确定 |
| 个别中文资料 | Shared Memory（共享内存）的简称，规范写法为 **SMEM** | 备选 |

**SM 访存** = GPU 通用线程的访存路径：指令 → SM 内 LD/ST 单元 → 寄存器堆 / Shared Memory / L1 → L2 → HBM（DRAM）。

### 8.2 TS

无统一标准定义，按资料来源语境：

| 语境 | TS 的可能含义 |
| --- | --- |
| GPU / 张量计算资料 | **Tensor Store（张量存储）**：张量数据在片上的存储与访存通路；或指 Tensor Core 的**数据搬运路径**（Hopper 起官方机制名为 **TMA**，Tensor Memory Accelerator，HBM→SRAM 异步直连搬运、不经过寄存器） |
| NVDLA 软件/编译器资料 | **Tensor Semantics（张量语义）**：描述每个张量在内存中的布局格式 |
| 通用 AI 芯片论文 | Tensor Store / Tensor Slice（张量切片），各家自创缩写，无标准 |

**TS 访存** = 张量数据的访存路径：Tensor Core 输入输出数据的专用搬运通路（异步、绕过寄存器、直达片上 SRAM）。

### 8.3 与 NVDLA 的关系

- NVDLA 内**没有 TS/SM 模块**；最接近的访存概念是 **CBUF/CMEM**（由 CDMA/BDMA 驱动搬入搬出）+ **CSC 调度**对 CBUF 的读写。
- 若在 NVDLA 资料中遇到这两个词，大概率是文章作者的自创缩写或引用 GPU 概念，需以原文上下文为准。

---

## 9. 待办 / 下一步

- [ ] 克隆 nvdla/hw 仓库到本地，用 project-reader 深度分析 RTL 结构
- [ ] 精读 Primer + Unit Description，补充各模块寄存器/数据流细节
- [ ] 明确验证目标：自研 NPU（NVDLA 作模板）还是移植 NVDLA 本身（vp_awsfpga 路线）
- [ ] FPGA 板卡资源预算（DSP/BRAM 容量 vs 目标配置）
- [ ] 调研验证环境选型：UVM 复用 vs 自建 C model 对比链路
