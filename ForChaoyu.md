# MoE-DeepEP

> 一个从零手写、profile 驱动的 MoE 专家并行项目，最终回答："DeepEP 到底解决了 MoE 的什么问题"。
> 这份 README 是写给我自己的——记录我做了什么、学到了什么、想明白了什么、还能往哪走。
> 数字都是真实跑出来的（单卡 RTX 5090 / 云端 2×H100），不是抄来的结论。

---

## 0. 我为什么做这个项目

我 CUDA / vLLM / profiling 的底子还行，但**从没真正碰过 MoE**。与其读十篇 DeepEP 的解读，不如自己把
一层 MoE 从单卡算子一路写到多卡通信，再去撞 DeepEP，看它到底比我强在哪、强在哪个场景。

目标不是"做出一个快的 MoE"，而是**用自己的代码和数据，把"DeepEP 为什么存在"这件事彻底想透**。

---

## 1. 项目结构

```
src/
  phase0_moe_literacy/      # MoE 入门：抓真实路由、量化负载倾斜
    capture_routing.py        # monkeypatch route_tokens_to_experts 抓真实 topk
    analyze_load.py           # 负载分布统计（CV、热/冷专家）
    inspect_moe.py            # 用 inspect.getsource 摸清 DeepSeek-V2-Lite 路由在哪
  phase1_single_gpu_moe/    # 单卡专家计算：从朴素到融合 Triton kernel
    common_moe.py             # ExpertWeights / 输入构造 / 单专家 FFN
    moe_layer_naive.py        # 逐专家 for 循环（基准）
    moe_layer_optimized.py    # torch 向量化（argsort/bincount/cumsum）
    moe_layer_bmm.py          # padding-to-max + bmm（反面教材，OOM）
    grouped_gemm_triton.py    # 单投影 grouped GEMM（v3: 预算 tile_expert + autotune）
    grouped_gemm_fused.py     # gate+up+SILU 融合 kernel + autotune
    moe_layer_triton.py       # 用 grouped GEMM 拼完整 SwiGLU
    moe_layer_triton_fused.py # 融合版 + CUDA Graph 封装
    bench.py                  # 六方对拍 + 测速
    probe_triton.py / roofline_check.py / profile_nsys*.py / graph_compare.py  # 各种探针
  phase2_expert_parallel/   # 多卡专家并行：dispatch/combine + 对标 DeepEP
    ep_reference.py           # 单进程语义基准（定义"正确"）
    ep_dist_naive.py          # 真多进程 3-all2all（gloo/nccl 双后端）
    ep_dist_opt.py            # 2-all2all（group-sum 推 recv_counts）
    expert_compute_fused.py   # 接 Phase1 融合 kernel 做本地专家计算
    ep_dist_fused.py          # 2-all2all + 融合 kernel 完整层
    ep_dist_overlap.py        # 异步 all2all + chunk 流水线重叠
    bench_cloud.py            # 云端总入口：串行 vs 重叠 vs DeepEP
    deepep_baseline.py        # 真 DeepEP 对标（按实测 API 适配）
    check_deepep_api.py       # 跑对标前核对 DeepEP 接口签名
docs/
  design_notes.md           # 全程作战记录（最重要，随做随记的踩坑+认知）
  01_why_deepep.md          # 最终结论文档
  cloud_setup_runpod.md     # 云端操作清单
```

> 待清理：`phase2` 里的 `dispatch_combine_naive.py` / `dispatch_combine_optimized.py` / `overlap.py`
> 是早期搭骨架的占位，真实现已在 `ep_dist_*.py`，可删。

---

## 2. 我做了什么（三个阶段）

### Phase 0 — 先搞懂 MoE 在干嘛
- 加载 DeepSeek-V2-Lite（E=64, top-6, H=2048），8-bit，**不用 trust_remote_code**（新 HF 原生支持）。
- 踩坑链：路由不在 `gate.forward`（gate 只是个 Linear），真正的 topk 在
  `mlp.route_tokens_to_experts`；forward hook 在 accelerate offload 下失效 → 改 monkeypatch 那个方法。
- **实测真实负载倾斜**：aggregate max/mean ≈ 10.6，CV ≈ 2.65，~90% 是冷专家。
  每一层都倾斜，不是个别现象。
- **想明白的事**：训练的 balance loss 只保证"训练分布上的聚合均衡"，**不保证单条输入推理时的
  逐 batch 均衡**。这就是为什么 MoE 的 all-to-all 难——buffer 必须按最坏情况（热专家）预留。

### Phase 1 — 单卡把专家计算做快
六个版本逐步演进，每一步都用数据逼出下一步：

| 版本 | 16384 token | 干掉了什么 | 引入了什么新问题 |
|------|-------------|-----------|----------------|
| naive | 1.00x | — | 倾斜→大量 32×32 小 GEMM 喂不饱 tensor core |
| torch_vec | 1.18x | 路由查找开销 | 专家 GEMM 仍逐专家 launch |
| torch_bmm | OOM | 小 GEMM（变 3 次大 bmm） | padding 到最长段，浪费 ~87% + 显存爆 |
| triton | 0.93x | 小 GEMM + padding | v1 kernel 内 O(E) 扫描，大输入拖慢 |
| **fused** | **1.28x** | 重复读 x + 独立 silu kernel | — |
| fused+graph | 1.23x | （想消空泡） | 无空泡可消，反被 replay copy 拖累 |

关键技术点：
- **grouped GEMM**：切块单位是"行 tile"不是"专家"，热点段自动多占 tile、冷门段占 1 个 tile，
  既无小 GEMM 也几乎无 padding（浪费从"对齐最长段 3000 行"降到"对齐 tile 边界 <64 行"）。
- **去掉 kernel 内 O(E) 扫描**：host 端用 searchsorted 预算 `tile_expert`，kernel O(1) 查表。
- **融合**：gate+up 读一遍 x 同时算两投影，SILU 进 epilogue（寄存器里算完，不单独起 kernel）。
- **autotune**：搜 BLOCK_N/K/stages/warps（BLOCK_M 固定 64，因 tile_expert 依赖它）。

### Phase 2 — 多卡专家并行 + 对标 DeepEP
- `ep_reference` 单进程模拟 dispatch/expert/combine，对拍"不分卡直算" **max_diff=0** ——
  证明分卡+通信数学等价于单卡。
- 真多进程：NCCL 要求一卡一 rank；本地单卡验证逻辑用 **gloo**（CPU 通信、可多 rank 共卡）。
- **3→2 all2all**：发"每专家行数"细账单，recv_counts 用 `view(ws,E_local).sum(1)` 本地推出，
  逐 token expert_id 用 repeat_interleave 本地生成，省掉第 3 次 all2all。
- 接 Phase1 融合 kernel 做本地专家计算，3 组随机（含非均匀权重）对拍 ALL PASS。
- 异步重叠：chunk 流水线 + async all2all，GPU 算 chunk_i 时 chunk_(i+1) 通信后台飞。
- 云端 2×H100 对标真 DeepEP。

---

## 3. 我学到 / 想明白了什么（最值钱的部分）

### 三个反直觉的发现（比"加速 X%"重要得多）

**① padding 消灭小 GEMM，反而慢 8 倍还 OOM。**
torch_bmm 把每段 padding 到最长段，倾斜越重浪费越大（~87% 在算 0），大输入直接撑爆显存。
教训：消灭一个问题（小 GEMM）若引入更大的问题（算空气 + 存空气），就是负优化。

**② CUDA Graph 不是"加了就快"——收益 = 空泡占比。**
融合后只剩 2 个饱满大 kernel，本就几乎无 kernel 间空泡，graph 无收益，还因 replay 的固定
buffer copy 略亏。graph 的主场是"kernel 多而碎"（如 decode 阶段 batch=1 的几微秒小 kernel）。

**③ 通信-计算重叠在单机 NVLink 是负优化。**
重叠提升随 token 单调升至 0.96 但**永不破 1**；随 chunk 单调降、无甜点。根因：单机 NVLink 带宽
极高，单次通信仅占总时间 ~10%，没多少延迟可藏；而切 N chunk → NCCL per-call 固定开销 ×N。
nsys 看到的就是：kernel 极小、gap > kernel，GPU 在 gap 里**空泡（等 CPU/NCCL 发起下一个通信）**。

### 关于 DeepEP（项目的最终答案）
DeepEP 单机**连初始化都过不去**（`Unable to dlopen libibverbs`）——它的 low_latency 路径强制
走 NVSHMEM/IBGDA，需 IB 网卡。这不是 bug，是它的**设计前提**：

> DeepEP 解决的是**多机跨节点、小消息、高频** all-to-all 的延迟瓶颈。此时单次通信小、次数多，
> NCCL 每次调用的固定启动开销成为主导，跨节点带宽又远低于 NVLink。DeepEP 用 IBGDA（GPU 直接
> 发起 RDMA、不经 CPU）+ 零 SM 占用 hook，把"发起一次小通信"的固定开销压到接近零——于是细粒度
> 重叠才有效、小消息才不被启动开销吞掉。

我的实测三件事共同指向它：单机通信 1.44ms 占比低 → 重叠负优化 → DeepEP 单机跑不起来。
**单机不是它的战场。我用"自己做不快 + DeepEP 跑不起来"反证了它的价值边界。**

### 方法论层面的肌肉
- **profile 分工**：nsys 通查找瓶颈 kernel（听诊器）；ncu 只深挖单个目标 kernel（显微镜）；
  roofline 在上 nsys 前先定性 compute/memory-bound。（我曾错用 ncu 扫全程，慢且无重点。）
- **怎么读 nsys 找空泡**：kernel 行连成密片 = 好；带缝 = 每条缝是 GPU 等 CPU 的空泡。
- **预测要用数据验证**：这个项目里我（和 AI 搭子）多次预测被数据推翻（重叠会加速、memory-bound、
  graph 有用……）。教训：猜测便宜、profile 也便宜，别用猜代替测。

---

## 4. 还能更进一步做什么（留给未来的我）

### 直接能补的
- [ ] **多机 + IB/RoCE 环境正面对标 DeepEP**：这是把"反证"变"正证"的关键。预期能看到 DeepEP
      在跨节点小消息下相对 NCCL 的显著优势，以及重叠从负优化转正。这是整个项目最该补的一块。
- [ ] **occupancy 根治**：融合 kernel ncu 实测 occupancy 仅 ~8.33%（算力却 80%+），autotune 只
      部分缓解。BLOCK_M 被固定（因 tile_expert 依赖它）限制了搜索。可尝试让 tile_expert 随
      BLOCK_M 动态重算，打开 BLOCK_M 搜索空间。
- [ ] **清理 phase2 的 3 个 stub 文件**，让 repo 干净。

### 想深入的方向
- [ ] **FP8 dispatch**：DeepEP 默认 use_fp8=True。实现 FP8 量化的 dispatch，看通信量减半后
      重叠的盈亏平衡点怎么移动。
- [ ] **真正的细粒度重叠**：当前重叠是 chunk 级。DeepEP 的 hook 是 kernel 级零 SM 占用，
      可以研究怎么用 CUDA stream + event 做到更细、且不增 per-call 开销。
- [ ] **负载均衡的影响**：Phase0 测出 ~10× 倾斜。可以做实验：人为改变倾斜程度，看 dispatch 的
      buffer 浪费、热专家 GEMM 的 tile 利用率怎么变——把 Phase0 的观察和 Phase1/2 的性能打通。
- [ ] **接真模型端到端**：现在专家计算用的是随机权重。把 DeepSeek-V2-Lite 的真实专家权重接进来，
      跑真实 forward，验证整层 numerics 对得上 HF 参考。

### 如果要写成对外内容
- 三个反直觉发现（padding 慢 8 倍 / graph 无收益 / 重叠负优化）每个都能单独成一篇有记忆点的帖子。
- "我用跑不起来 DeepEP 反证了它为什么存在" 是个好标题。

---

## 5. 复现命令速查

```bash
# Phase 1 六方对比
python -m src.phase1_single_gpu_moe.bench --synthetic --tokens 16384 --skew 1.5

# Phase 1 各种探针
python -m src.phase1_single_gpu_moe.probe_triton       # 拆解耗时 + 扫 BLOCK
python -m src.phase1_single_gpu_moe.roofline_check     # compute/memory-bound 定性
python -m src.phase1_single_gpu_moe.graph_compare      # graph 在碎/大 kernel 的对比

# Phase 2 本地验证（gloo 单卡多进程）
SINGLE_GPU=1 torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_fused
SINGLE_GPU=1 torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.ep_dist_overlap

# Phase 2 云端（真双卡 NCCL）
torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud --tokens 4096 --chunks 4
# 扫 token / chunk 验证重叠盈亏
for t in 2048 8192 32768 65536; do torchrun --nproc_per_node=2 -m src.phase2_expert_parallel.bench_cloud --tokens $t --chunks 4; done
```

---

## 6. 关键实测数字存档（2×H100, NVLink, 单机, 4096 token/rank）

- 自研纯通信 dispatch+combine：**1.443 ms**
- 单次 dispatch（2 all2all）：**0.656 ms**
- 串行完整 MoE 层：**3.70 ms**
- 重叠提升：扫 token 0.47→0.96（永不破 1）；扫 chunk 0.99→0.75（越切越差）
- DeepEP low_latency：**单机无法初始化（需 IB 网卡）**

> 详细推导和踩坑全程见 `docs/design_notes.md`，最终结论见 `docs/01_why_deepep.md`。