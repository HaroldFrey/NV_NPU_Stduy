# CUDA 软件栈概念笔记（CUDA / PTX / SASS / CUBIN）

> 学习目的：扫清 NVIDIA GPU 软件栈核心术语障碍，能读懂 GPU/NPU 部署与性能调优类文章
> 创建日期：2026-08-12
> 背景：在查 DGX Spark（GB10）资料时遇到 `sm_121`、`cubin`、PTX 前向兼容等词，整理成文

---

## 0. 一条编译流水线（总览）

```
CUDA C/C++ 源码 ──nvcc──▶ PTX（虚拟指令集） ──ptxas──▶ SASS（机器码） ──打包──▶ CUBIN ──▶ GPU 执行
     (你写的代码)          (架构无关中间层)              (特定 GPU 真实指令)     (二进制文件)
```

CUDA / PTX / SASS / CUBIN 不是并列的四个东西，而是**同一条编译流水线上的四个节点**：
源码 → 中间表示 → 机器码 → 二进制产物。

---

## 1. CUDA —— 整个生态的名字

**CUDA = Compute Unified Device Architecture**（统一计算设备架构），包含三样东西：

| 组成 | 内容 | 要点 |
|---|---|---|
| 编程模型 | 异构计算、线程层次、存储层次 | 异构：CPU=host（发指令），GPU=device（跑 kernel） |
| 语言扩展 | CUDA C/C++ | 就是 C++ 加 `__global__`、`threadIdx` 等扩展 |
| 工具链 | `nvcc` 编译器、runtime/driver API、cuBLAS/cuDNN 等库 | 完整的开发+运行环境 |

**编程模型核心概念**：

```
grid（网格，一个 kernel 调用）
 └── block（线程块）── 可同步、可共享 SMEM
      └── warp（束）= 32 个线程，SIMT 执行的最小调度单位
           └── thread（线程）
```

- 执行模式是 **SIMT**（单指令多线程）：一个 warp 同时执行同一条指令、处理不同数据
- 存储层次：寄存器堆 → Shared Memory（SMEM）→ L1 → L2 → HBM（片外）

---

## 2. PTX —— 中间桥梁

**PTX = Parallel Thread eXecution**（并行线程执行）。

- **类似汇编，但绑定的是"虚拟 GPU"**，不针对任何具体硬件型号
- 地位相当于 **LLVM IR**：架构无关的中间表示
- 由 `nvcc` 把 CUDA C 编译产生，`ptxas` 再把它编译成真实机器码

**它存在的意义**：程序只要带一份 PTX，运行时驱动可以 **JIT（即时编译）成当前硬件的 SASS**——这就是"前向兼容"机制的来源。老程序带 PTX，就能在新架构 GPU 上自动获得支持。

---

## 3. SASS —— GPU 的真实机器码

**SASS = Streaming ASSembly**（流汇编）。

- **每个 GPU 架构（compute capability）有自己的 SASS 指令集，互不通用**
  - A100（sm_80）的 SASS 不能在 sm_121（GB10）上直接跑
- 是硬件真正执行的指令：FMA 计算、LDG/STG 访存、MMA 张量指令等
- 不面向人类阅读，反汇编用 `nvdisasm` / `cuobjdump`
- **性能调优的最终依据**：寄存器分配、指令调度、访存指令优化、spill（寄存器溢出）都要看 SASS 层

---

## 4. CUBIN —— 编译产物

**CUBIN = CUDA Binary**（CUDA 二进制）。

- 针对**特定 sm 架构**的二进制文件，包含 SASS + 元数据（对应文件后缀 `.cubin`）
- 实际交付时多个架构的 cubin 打包进 **`.nv_fatbin`**（fat binary，"一包多架构"）
- 运行时驱动**按当前 GPU 的 sm 挑选匹配的 cubin**：
  1. 找到匹配 cubin → 直接加载执行
  2. 无匹配但带 PTX → JIT 编译 PTX
  3. 都没有 → 报错

---

## 5. Compute Capability（sm_XX）—— 理解一切的钥匙

GPU 架构代际标识，直接决定 SASS 指令集与硬件特性：

| sm | 架构 | 代表芯片 |
|---|---|---|
| sm_80 | Ampere | A100 |
| sm_90 | Hopper | H100 |
| sm_100 | Blackwell（数据中心） | B200 / GB200 |
| sm_120 | Blackwell（消费级） | RTX 50 系 |
| sm_121 | Blackwell（GB10） | **DGX Spark** |

**为什么重要**：预编译库（PyTorch、vLLM、Triton 等）只认特定 sm 的 cubin。之前查到的 DGX Spark 兼容性问题根源就在这：镜像只编了 sm_100 的 cubin，就在 sm_121 上拒绝加载；解法是用 sm_120 的 PTX 前向兼容或源码重编。

---

## 6. 实际场景（为什么你会遇到这些词）

1. **部署兼容性**：跑预编译的 AI 库时，cubin 与当前 GPU 架构不匹配 → JIT 兜底或直接报错
2. **性能调优**：反汇编 SASS 看指令级细节（访存指令、寄存器压力、是否 spill）
3. **编译参数**：`nvcc -arch=sm_121` 等参数决定编出什么架构的 cubin/PTX

---

## 7. 用 FPGA 的话说（类比表）

| NVIDIA 侧 | FPGA 侧类比 |
|---|---|
| CUDA C 源码 | RTL（行为描述） |
| PTX | 综合后的中间网表（架构无关） |
| SASS | 目标器件的机器码（LUT/FF/DSP 布线结果） |
| CUBIN | 编译好的二进制（"可执行"形态） |
| compute capability sm_XX | 器件系列（Artix-7 vs UltraScale+） |
| JIT：PTX→SASS | 按目标器件做的适配编译 |

**启示**："中间表示 + 多目标适配"的软件栈设计，和 NPU 验证时"一套测试激励、多套目标适配"的思路相通。

---

## 8. 工具速查

| 工具 | 作用 |
|---|---|
| `nvcc` | CUDA C → PTX / CUBIN 的编译器 |
| `ptxas` | PTX → SASS 的汇编器（nvcc 内部调用） |
| `cuobjdump` | 查看可执行文件 / fatbin 里打包了哪些 cubin |
| `nvdisasm` | 把 SASS 反汇编成可读指令流 |
| `nvidia-smi` | 查看当前 GPU 架构（compute capability）、驱动、显存 |
