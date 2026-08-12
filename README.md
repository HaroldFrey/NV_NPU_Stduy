# NV_NPU_Stduy — NVDLA（NVIDIA NPU）架构学习笔记

> FPGA 工程师视角的 NVIDIA NPU 架构学习资料库，目标是为**搭建 NPU FPGA 原型验证系统**提供参考架构。

## 学习对象

**NVDLA（NVIDIA Deep Learning Accelerator）** —— NVIDIA 2017 年开源的深度学习推断加速器，基于 Xavier SoC DLA，RTL 全开源（[github.com/nvdla/hw](https://github.com/nvdla/hw)，含 RTL + C model + UVM testbench）。

它也是唯一有官方 FPGA 移植（AWS F1 / Xilinx 端口、FireSim 集成）的 NV 推断加速器，是"NPU 的教科书级架构"。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| [NVDLA架构学习笔记.md](NVDLA架构学习笔记.md) | 架构深度分析：总体架构 / 五级卷积流水线 / 后处理分区 / 总线与编程模型 / 关键参数 / **FPGA 原型验证启示** |
| [NVDLA架构图解.html](NVDLA架构图解.html) | 可视化架构图（浏览器打开，自动适配明暗主题）：SoC 集成视图 / 内部模块架构 / 单层卷积数据流 / CMAC 阵列 |
| [CUDA软件栈概念笔记.md](CUDA软件栈概念笔记.md) | CUDA / PTX / SASS / CUBIN 概念整理（编译流水线、compute capability、FPGA 类比） |
| [NVIDIA工具链与自研NPU验证笔记.md](NVIDIA工具链与自研NPU验证笔记.md) | 自研 NPU + 兼容 NVIDIA 工具链的验证方法：全链路、兼容层次、检查点、golden 比对 |
| [NPU与GPU概念笔记.md](NPU与GPU概念笔记.md) | NPU 与 GPU 的本质区别：定义、核心对比、边界模糊化趋势、对验证工作的意义 |

## 核心结论（速览）

- **架构本质**：CPU 配置、流水线算、大缓冲省带宽的专用推断协处理器
- **双总线**：CSB（配置，慢）+ DBB/AXI4（数据，快）彻底分离
- **五级卷积流水线**：CDMA → CBUF(512KB) → CSC → CMAC(2048×INT8) → CACC → SDP/PDP/CDP（共用 CMEM 512KB）
- **验证方法学**：C model 黄金参照 + UVM 验证环境 + 软硬件协同（compiler/KMD/UMD），NPU 验证与普通 IP 验证的本质区别

## 学习路线（进行中）

- [x] NVDLA 架构分析（笔记 + 图解）
- [x] GPU 软件栈概念扫盲（CUDA/PTX/SASS/CUBIN）
- [ ] 克隆 nvdla/hw 仓库，深度分析 RTL 结构
- [ ] FPGA 板卡资源预算（DSP/BRAM 容量 vs 目标配置）
- [ ] 明确验证目标：自研 NPU（NVDLA 作模板）还是移植 NVDLA 本身

## 参考资料

- [NVDLA Primer（官方白皮书）](https://nvdla.org/primer.html)
- [Unit Description](https://nvdla.org/hw/v1/ias/unit_description.html)
- [NVDLA HotChips 30 演讲](https://old.hotchips.org/hc30/2conf/2.08_NVidia_DLA_Nvidia_DLA_HotChips_10Aug18.pdf)
- [nvdla/hw 仓库](https://github.com/nvdla/hw)（RTL + C model + testbench）
- [nvdla/vp_awsfpga](https://github.com/nvdla/vp_awsfpga)（AWS FPGA 虚拟平台）
