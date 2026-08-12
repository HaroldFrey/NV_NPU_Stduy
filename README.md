# NV_NPU_Stduy — H100（Hopper）架构与自研 NPU 验证学习笔记

> FPGA 原型验证工程师视角的学习资料库。
> **项目背景**：自研 NPU 参考 **H100（Hopper）架构**、兼容 **NVIDIA 工具链**（CUDA/TensorRT），测试 case 走 NVIDIA 编译链路，在 FPGA 原型上验证、与 GPU golden 比对。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| [H100架构解析笔记.md](H100架构解析笔记.md) | H100 架构深度解析：SM / Tensor Core / WGMMA / TMA / 访存层次，及对自研 NPU 的启示 |
| [NVIDIA工具链与自研NPU验证笔记.md](NVIDIA工具链与自研NPU验证笔记.md) | 自研 NPU + 兼容 NVIDIA 工具链的验证方法：全链路、兼容层次、检查点、golden 比对 |
| [NPU与GPU概念笔记.md](NPU与GPU概念笔记.md) | NPU 与 GPU 的本质区别、边界模糊化趋势、对验证工作的意义 |
| [CUDA软件栈概念笔记.md](CUDA软件栈概念笔记.md) | CUDA / PTX / SASS / CUBIN 编译流水线概念整理（含 FPGA 类比） |

> 📌 NVDLA 学习笔记与图解已移出本仓库（NVDLA 不是本 NPU 的参考对象），归档至 [FPGA_SOC_Verify/NVDLA](https://github.com/HaroldFrey/FPGA_SOC_Verify) 作为开源参考项目资料。

## 核心结论（速览）

- **参考对象**：H100（Hopper）——AI 计算部分 = Tensor Core（WGMMA）+ TMA 异步搬运 + SMEM 数据复用
- **验证链路**：测试 case 经 NVIDIA 编译链（CUDA→PTX→SASS/CUBIN）→ 自研翻译层 → FPGA 原型 → GPU golden 比对
- **关键指令语义**（翻译层验证对象）：**WGMMA**（M64×N×K16 矩阵乘）/ **cp.async.bulk**（TMA 搬运）/ **mbarrier**（同步）

## 学习路线（进行中）

- [x] H100 架构解析（笔记）
- [x] NPU/GPU 概念 + CUDA 软件栈扫盲
- [x] 工具链与自研 NPU 验证方法
- [ ] Hopper 指令集（PTX）逐指令解析——翻译层的语义清单
- [ ] 兼容层次确认（API 级 / PTX 中间表示级 / SASS 指令级）
- [ ] FPGA 板卡资源预算与吞吐目标

## 参考资料

- [NVIDIA H100 GPU Whitepaper（官方，填表单获取 PDF）](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Deep Dive on the Hopper TMA Unit（PyTorch）](https://pytorch.org/blog/hopper-tma-unit/)
- [PTX ISA 文档（wgmma / cp.async.bulk / mbarrier）](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS（WGMMA/TMA 参考实现）](https://github.com/NVIDIA/cutlass)
