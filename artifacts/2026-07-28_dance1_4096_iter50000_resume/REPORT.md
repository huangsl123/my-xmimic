# DexEVT dance1：4096 环境、50,000 次续训、仿真与可视化

日期：2026-07-29  
任务：`Tracking-Flat-DexEVT-Wo-State-Estimation-v0`  
正式运行：`2026-07-28_15-42-13_dance1_4096_additional_50000_from_model9_fixed`  
续训起点：`2026-07-28_13-37-30_dance1_4096_smoke_fixed/model_9.pt`  
最终 checkpoint：`model_50008.pt`

> 本报告中的动态数值均来自最终 TensorBoard、逐步回放 CSV 和独立验证 JSON。

## 结论

本次以已修复 motion 身体名称映射的 `model_9.pt` 为起点，在 4096 个并行环境中执行 **50,000 次新的 PPO 更新**。每次更新采集 `4096 × 24 = 98,304` 个 transition，正式续训段共 **4,915,200,000** 个 transition；连同 smoke 阶段实际执行的 10 次更新，整条训练链路共处理 **4,916,183,040** 个 transition。

训练于 `2026-07-29 05:24:21 +08:00` 正常退出（exit code 0），墙钟耗时约 `13 h 42 min 12 s`。TensorBoard 记录 step `9..50008`，共 `50,000` 条连续且不重复的更新记录；最终保存 `model_50008.pt`。共 37 个 TensorBoard 标签、1,849,998 个标量点，未发现 NaN/Inf。

末尾 100 次更新的 mean reward 为 `22.2874`，mean episode length 为 `499.6794 / 500`。固定 seed 42 的洁净长回放完成全部 3972 帧且无终止，但严格位置精度门槛未通过；动作完成率由 smoke 基线的 **3.07%** 变为 `100.00%`。3 次启用训练期随机事件与观测扰动的长回放都完整完成动作，但 `0/3` 通过严格位置验收。

## 验收结果

| 验收项 | 标准 | 结果 |
|---|---|---|
| 训练完整性 | 正常退出；step=50008；存在且可加载 `model_50008.pt` | **通过** |
| 数值稳定性 | TensorBoard 和仿真全部数值 finite | **通过** |
| 固定种子仿真 | 恰好 3972 步、无 termination、全部数值 finite | **通过** |
| 平台位置 | 起点半径 `<0.05 m`；最大中心距离 `<1.0 m`；XY drift P95/max `<0.25/0.5 m` | **未通过**：起点通过；1.135 m、0.593/0.651 m 超限 |
| 视频 | 25 FPS、3972 帧、158.88 s、1280×720、完整解码、非黑且非冻结 | **通过** |
| 扰动长回放 | seeds 0/1/2 恰好 3972 步、无 termination、finite，并通过上述位置阈值 | **基础完成 3/3；严格位置 0/3** |
| ONNX | checker、形状、normalizer、元数据和参考推理通过 | **通过** |

## 数据与身体映射

训练使用 [dance1_easy_named.npz](../../motion_data/dance1_easy_named.npz)，SHA-256：

```text
5c9a59f6ae003df62313bd5a41ec9bb6f4626cca90819a8841cdde9f062127ed
```

| 项目 | 值 |
|---|---:|
| Motion 帧数 / 频率 | 3972 / 100 Hz |
| 按样本数计算的时长 | 39.72 s |
| 首帧至末帧时间跨度 | 39.71 s |
| Joint / motion body / runtime tracking body | 23 / 39 / 24 |
| 数值有限性 | 所有数值数组均 finite |

该文件与 `motion_example/dance1_easy.npz` 的数值数组逐元素一致，只增加 `body_names` 和 `body_layout`。运行时 24 个跟踪 body 均按名称映射，不再用 24-body articulation 的数字索引误读 39-body motion。独立 URDF 正向运动学复核的首帧位置最大误差为 `3.218e-7 m`，平均误差为 `1.586e-7 m`。

历史错误会让运行时左脚踝索引 18 错读成 `camera_body_front_link`，制造虚假的脚部高度误差并频繁触发 `ee_body_pos`。当前加载链路会拒绝：`body_names` 缺失且 body count 与 runtime 不一致、重复名字、缺少 tracked body、索引越界，以及 motion FPS 与控制频率不符。`body_layout` 当前只作为元数据保存，并未单独校验。尚存的 schema 局限是 NPZ 没有 `joint_names`，23 维 joint 数组仍依赖既定顺序。

## 训练配置与数据口径

| 项目 | 值 |
|---|---:|
| 并行环境 | 4096 |
| 正式续训更新数 | 50,000 |
| 每环境每次更新步数 | 24 |
| 每次更新 transition | 98,304 |
| 正式续训 transition | 4,915,200,000 |
| Smoke + 正式训练 transition | 4,916,183,040 |
| Seed | 42 |
| Physics dt / decimation / control rate | 0.005 s / 2 / 100 Hz |
| Episode 上限 | 5 s / 500 控制步 |
| 每环境续训控制步 / 模拟时间 | 1,200,000 / 12,000 s |
| 4096 环境累计模拟时间 | 49,152,000 s，约 568.889 天 |
| Actor / critic observation | 124 / 346 |
| Action | 23 维；scale 0.25 |
| Actor / critic MLP | 512, 256, 128；ELU |
| PPO | 5 epochs；4 minibatches；clip 0.2 |
| Gamma / lambda / entropy / desired KL | 0.99 / 0.95 / 0.005 / 0.01 |
| Empirical normalization | 开启 |
| Actuator delay | 0–4 控制步，约 0–40 ms |
| Raw policy action clip | `[-100, 100]` |
| GPU | NVIDIA GeForce RTX 5090，32607 MiB |
| 运行环境 | Isaac Sim 5.0；Python 3.11 |

任务名虽然包含 `Flat`，实际保存配置使用 5×10 个 8 m×8 m tile：约 50% 平地、25% 轻微噪声、25% 缓坡。因此这是**混合地形训练**，不是纯平地训练。

Wo-State actor 不接收 `motion_anchor_pos_b` 和 `base_lin_vel`，critic 仍使用 privileged observation。训练开启 actor observation corruption，并随机化质量、COM、摩擦、关节参数、控制增益、动作延迟及 1–5 s 随机 push。

Motion 长 39.72 s，但每个训练 episode 最多 5 s。每次 reset 随机选择 motion 相位，再训练至多 5 s 的连续片段；完整 3972 帧能力必须由独立长回放验证。

Policy 输出先由 wrapper 截断到 `[-100,100]`，action manager 再执行 `q_target = q_default + 0.25 × raw_action`。ONNX 只包含 actor、normalizer 与 motion reference，不包含 wrapper clip 或机器人侧的 default-offset/scale 执行流程；部署端必须按 metadata 中的 joint 顺序和 action scale 复现该逻辑。

正式训练命令：

```bash
python -u scripts/rsl_rl/train.py \
  --task Tracking-Flat-DexEVT-Wo-State-Estimation-v0 \
  --num_envs 4096 --max_iterations 50000 --device cuda:0 \
  --motion_file motion_data/dance1_easy_named.npz --headless \
  --logger tensorboard --seed 42 --resume True \
  --load_run 2026-07-28_13-37-30_dance1_4096_smoke_fixed \
  --checkpoint model_9.pt \
  --run_name dance1_4096_additional_50000_from_model9_fixed
```

## 续训与 checkpoint 语义

源 checkpoint `model_9.pt` 的 SHA-256 为：

```text
05e0498ad0999d09beff490e99c5a05b3e3c2e9dd5c886fbddb73a8f9497e581
```

其内部 `iter=9`。当前 runner 从 `range(9, 50009)` 执行恰好 50,000 次更新，所以本次标签为 9–50008，标签 9 相对 smoke 重复一次。因此：

- 最终模型必须使用 `model_50008.pt`；
- `model_50000.pt` 仍少最后 8 次更新；
- smoke 实际执行 10 次，续训再执行 50,000 次，共 50,010 次 PPO rollout/update 迭代；
- 新 run 的 TensorBoard 横轴从 9 开始，不能把 step 50008 当成实际更新总数。

Resume 恢复 policy、value function、optimizer、actor/critic empirical normalizer 和迭代标签；源 optimizer learning rate 为 `0.0002562890625`，policy noise std 均值约 0.90，normalizer count 为 983,040。环境状态、RNG、rollout buffer、`tot_timesteps` 和 `tot_time` 不恢复；新 run 使用 `init_at_random_ep_len=True`，首批 episode length 会随机初始化，terrain generator 的 seed 也为 `null` 并依赖全局 RNG。因此它不是逐比特的物理环境续跑，开头曲线还包含 warm-start 统计偏差。

## 训练曲线

下表中的终止值是 RSL-RL 聚合日志，不是 4096 环境的原始事件计数。`anchor_pos` 实际只检查 root z 误差是否超过 0.25 m，`ee_body_pos` 只检查双脚 z 误差是否超过 0.25 m，`anchor_ori` 阈值为 0.8 rad。

| 指标 | 首值 / step 9 | 最后 100 均值 | 最后 1000 均值 | 最终值 / step 50008 |
|---|---:|---:|---:|---:|
| Mean reward | `-0.1243` | `22.2874` | `22.2773` | `22.3349` |
| Mean episode length | `16.7500` | `499.6794` | `499.6157` | `500.0000` |
| Anchor position error | `0.0698` | `0.1923` | `0.1889` | `0.1951` |
| Body position error | `1.1302` | `0.04044` | `0.04060` | `0.04044` |
| Joint position error | `1.4302` | `0.5860` | `0.5782` | `0.6018` |
| `anchor_pos` termination | `0.0000` | `0.00292` | `0.00392` | `0.0000` |
| `ee_body_pos` termination | `25.0833` | `0.00375` | `0.00433` | `0.0000` |
| Value loss | `0.02081` | `0.00229` | `0.00236` | `0.00252` |
| Throughput | `47,503` | `100,548` | `100,414` | `98,785` steps/s |

主要奖励分量在最后 100 次更新中的均值为：anchor position/orientation `0.7501 / 0.4835`，body position/orientation `0.9686 / 0.8778`，body linear/angular velocity `0.9021 / 0.8651`；action-rate 与 torque 惩罚为 `-0.2735 / -0.1106`。Policy noise std 从 `0.8928` 降至最终 `0.2452`；最终 surrogate loss 为 `-0.00922`，optimizer learning rate 为 `5.0625e-5`。

曲线可分为三个阶段：step 9–约 5,000 为快速学习期，reward 从负值升至约 20 且 episode length 接近 500；约 5,000–20,000 为缓慢改进期；约 20,000–50,008 为平台期，reward 稳定在约 22.2–22.4、value loss 约 0.0023、吞吐约 100k steps/s，末段未见发散。

### 训练可视化

- [训练总览](training_analysis/training_overview.png)
- [最后 1000 次更新](training_analysis/training_tail.png)
- [奖励分量](training_analysis/reward_components.png)
- [跟踪误差](training_analysis/tracking_errors.png)
- [完整标量 CSV](training_analysis/scalars.csv)
- [机器可读训练摘要](training_analysis/summary.json)
- [完整训练日志](training.log)

## 固定种子仿真与 smoke 基线

正式回放使用最终 checkpoint、seed 42、单环境，关闭训练期随机事件、observation corruption 和调试 marker。`--play_full_motion` 将相位固定为 0，以 motion 首帧重写 robot root/joint state，把 episode horizon 从训练时的 5 s 延长到约 39.73 s，并在末帧保持参考；所有非-timeout termination 仍启用。评估以 100 Hz 运行，在真实 `done` 时停止，不拼接自动 reset 后的回合。

| 指标 | Smoke `model_9.pt` | 最终 `model_50008.pt` |
|---|---:|---:|
| 已记录步数 / 仿真时长 | 122 / 1.22 s | `3972 / 39.72 s` |
| 动作完成率 | 3.07% | `100.00%` |
| 首次终止 | step 121，`anchor_pos` | 无终止 |
| Reward mean / P95 | 0.0436 / 0.0537 | `0.04377 / 0.05206` |
| Anchor position mean / P95 / max | 0.169 / 0.527 / 0.623 m | `0.3809 / 0.5934 / 0.6506 m` |
| Body position mean / P95 / max | 0.090 / 0.276 / 0.338 m | `0.04244 / 0.06343 / 0.08066 m` |
| Joint position mean / P95 / max | 0.531 / 1.006 / 1.099 | `0.5629 / 0.8046 / 1.0219` |
| XY tracking drift P95 / max | 0.491 / 0.574 m | `0.5932 / 0.6506 m` |
| 离 tile 中心最大距离 | 0.574 m | `1.1351 m` |

Isaac Lab 会在 terminal step 返回前自动 reset。误差与位置统计会排除 terminal 行，`logged_steps`、完成率和 termination 统计仍包含该行；因此 smoke 表中的误差是 121 个终止前稳定样本的统计，不是 CSV 最后一行 reset 后的姿态。

最终模型把可连续执行长度从 122 步提高到完整 3972 步，且没有任何 termination；body position P95 相比早停基线下降约 77.0%。不过完整轨迹后半段出现累积的 root XY 偏差，严格 P95/max 漂移门槛均未通过。因此结论是“完整动作能力显著修复，水平根节点跟踪仍需定向优化”，不能只用 100% 完成率宣称全指标合格。

### 平台中心

Tile 尺寸为 8 m×8 m，半宽 4 m。最终回放起点相对 env 0 tile 中心为 `(0.000656, -0.000038) m`，距离 `0.000657 m`；全程最大中心距离 `1.1351 m`，占半宽 `28.38%`。

参考动作自身末帧相对首帧约移动 `(0.458, -0.165) m`，水平净位移约 0.487 m。因此后续离开几何中心不一定是初始化错误；需要结合相对参考的 XY drift。Viewer 已绑定 `origin_type="env"`、`env_index=0`，不再把 world origin 误当成当前平台中心。

### 0.25 倍速视频

物理和策略仍以 100 Hz 运行，`playback_speed=0.25` 将视频编码为 25 FPS，不改变动力学。完整 3972 个控制步理论生成 158.88 s 视频。

- [0.25 倍速视频](final_evaluation_model_50008/nominal_seed42/video/rl-video-step-0.mp4)
- [12 帧关键画面](final_evaluation_model_50008/video_acceptance/contact_sheet_12.png)
- [视频解码记录](final_evaluation_model_50008/video_acceptance/decode_progress.txt)
- [视频验证 JSON](final_evaluation_model_50008/video_acceptance/summary.json)
- [视频解码错误日志](final_evaluation_model_50008/video_acceptance/decode_errors.log)
- [仿真分析图](final_evaluation_model_50008/nominal_seed42/analysis/playback_analysis.png)
- [逐步指标](final_evaluation_model_50008/nominal_seed42/metrics.csv)
- [仿真摘要](final_evaluation_model_50008/nominal_seed42/analysis/summary.json)

## 随机化鲁棒性

严格通过条件为：恰好 3972 步、无 termination、全部数值 finite、起点半径 `<0.05 m`、离 tile 中心最大值 `<1.0 m`、XY drift P95 `<0.25 m`、XY drift max `<0.5 m`。Anchor/Body P95 仅用于报告跟踪质量，当前 strict pass 没有为它们设置阈值。

启用随机事件的长回放会保留质量、COM、摩擦、关节参数、增益、延迟、interval push 和 actor observation corruption；但 full-motion 初始化会覆盖随机 reset 产生的 root/joint pose、velocity 和随机 motion phase。因此它是训练扰动的基础测试，不是完整重现训练 reset 分布。

| 场景 | 完成率 | 终止 | Anchor P95 | Body P95 | XY drift P95 / max | 严格通过 |
|---|---:|---|---:|---:|---:|---|
| Nominal seed 42 | `100.00%` | 无 | `0.5934 m` | `0.06343 m` | `0.5932 / 0.6506 m` | 否 |
| Randomized seed 0 | `100.00%` | 无 | `0.6350 m` | `0.05188 m` | `0.6346 / 0.7025 m` | 否 |
| Randomized seed 1 | `100.00%` | 无 | `1.3758 m` | `0.06717 m` | `1.3755 / 1.5028 m` | 否 |
| Randomized seed 2 | `100.00%` | 无 | `0.6423 m` | `0.05418 m` | `0.6413 / 0.6868 m` | 否 |

三个扰动 seed 均完整执行且无终止，最大中心距离为 1.028–1.170 m，仍远小于 4 m 平台半宽；body position P95 也保持在 0.052–0.067 m。但三者都因严格 XY drift 与中心距离门槛失败，seed 1 的漂移最明显。当前策略具备基础抗扰完成能力，却还没有达到高精度水平根节点稳健跟踪。

- [最终模型验收对比图](final_evaluation_model_50008/final_acceptance/evaluation_comparison.png)
- [最终模型验收 CSV](final_evaluation_model_50008/final_acceptance/evaluation_comparison.csv)
- [最终模型验收 JSON](final_evaluation_model_50008/final_acceptance/evaluation_comparison.json)
- [训练前后对比图](final_evaluation_model_50008/baseline_comparison/evaluation_comparison.png)

## ONNX

由 `model_50008.pt` 加载后重新导出的最终 ONNX 候选图为 `final_evaluation_model_50008/nominal_seed42/onnx/policy.onnx`，SHA-256 为 `eb6443e96af737df76f1859cf147400a686b34bb43168cba7f4ce66b5978e6e1`。训练进程是在导出器末帧修复前启动的，因此不使用 run 根目录的周期 ONNX 作为最终候选。验证结果：

- 输入：`obs [1,124]`、`time_step [1,1]`；
- 输出：23 维 action，以及 joint/body motion reference；
- 10 个 initializer，包含 `[1,124]` normalizer mean 和 divisor；
- joint names、stiffness、damping、default pose、observation names 和 action scale 元数据齐全；
- opset 11，固定 batch 1，float32 输入；
- actor/normalizer 与 checkpoint 逐数组一致，六组完整 motion 常量与命名 NPZ 一致；
- `onnx.checker`、全部常量 finite、时间边界与 ONNX reference inference 全部通过；
- 三组 actor 数值等价测试的最大绝对误差不超过 `7.153e-7`，六组 motion 常量逐元素最大差为 `0`。

该检查使用 ONNX 标准 `ReferenceEvaluator`，环境没有安装 ONNX Runtime；因此它证明图结构、来源和参考实现一致，不等同于目标部署 runtime 的兼容性认证。

- [ONNX policy](final_evaluation_model_50008/nominal_seed42/onnx/policy.onnx)
- [ONNX 验证摘要](final_evaluation_model_50008/onnx_validation/summary.json)

## 局限

1. 任务名含 `Flat`，实际训练为平地、轻噪声和缓坡混合地形。
2. 训练 episode 是随机相位的至多 5 s 片段，不是一次完整 39.72 s 动作。
3. Resume 不恢复 RNG 与环境状态，不是逐比特复现。
4. 新 run 从 step 9 开始，且标签 9 相对 smoke 重复。
5. `anchor_pos` 和 `ee_body_pos` 是 z-only 终止条件；未终止不等价于 3D 高精度。
6. 洁净回放与训练随机化分布不同，所以另做 3 个随机化 seed。
7. NPZ 已有 `body_names`，但仍缺 `joint_names`。
8. 参考动作本身含约 0.487 m 水平位移。
9. 训练基于 commit `fe294bc00f42fb1daaf8b0d340475d6ce308daea` 加未提交修改；run diff 不包含未跟踪文件正文。
10. 3 个随机化 seed 只能说明基础鲁棒性，不能等同于部署级统计保证。
11. ONNX 不内嵌 wrapper action clip 或机器人 action-manager 执行逻辑，部署端必须复现。
12. ONNX 验证器会检查 joint/body 名称数量、唯一性和 motion 可解析性，但不会替目标机器人验证其外部 canonical joint/body 顺序；部署前仍应与硬件接口逐项核对。

## 可复现性

| 文件 | SHA-256 |
|---|---|
| Motion NPZ | `5c9a59f6ae003df62313bd5a41ec9bb6f4626cca90819a8841cdde9f062127ed` |
| 起始 `model_9.pt` | `05e0498ad0999d09beff490e99c5a05b3e3c2e9dd5c886fbddb73a8f9497e581` |
| 最终 checkpoint | `e27c5aab5a6040414c464c38a4ba912f048be33fd76c6ebd0d7672c9d5b5696f` |
| `agent.yaml` | `a457d7fd70da14208c0cd85bbc151e9fc993a534b98477ec35140c5070518a3d` |
| `env.yaml` | `bdacf4dd8871ae0e6b1b29b1b30e936bb6377e5cf597c27ea6e14d566fcf9c4b` |
| 捕获的 `xMimic.diff` | `53770ec6ccab3f59f4c82c35499bd5711d95c80fd277a37153f43e43debb4d96` |
| 最终 ONNX | `eb6443e96af737df76f1859cf147400a686b34bb43168cba7f4ce66b5978e6e1` |

## 产物索引

- [训练运行目录](../../logs/rsl_rl/dex_evt_flat/2026-07-28_15-42-13_dance1_4096_additional_50000_from_model9_fixed/)
- [最终 checkpoint](../../logs/rsl_rl/dex_evt_flat/2026-07-28_15-42-13_dance1_4096_additional_50000_from_model9_fixed/model_50008.pt)
- [Agent 配置](../../logs/rsl_rl/dex_evt_flat/2026-07-28_15-42-13_dance1_4096_additional_50000_from_model9_fixed/params/agent.yaml)
- [Environment 配置](../../logs/rsl_rl/dex_evt_flat/2026-07-28_15-42-13_dance1_4096_additional_50000_from_model9_fixed/params/env.yaml)
- [代码差异快照](../../logs/rsl_rl/dex_evt_flat/2026-07-28_15-42-13_dance1_4096_additional_50000_from_model9_fixed/git/xMimic.diff)
- [机器人仓库差异快照](../../logs/rsl_rl/dex_evt_flat/2026-07-28_15-42-13_dance1_4096_additional_50000_from_model9_fixed/git/TienKung-Lab.diff)
- [最终训练快照](final_training_snapshot.json)
- [Checkpoint 独立验证](checkpoint_validation/summary.json)
- [训练分析目录](training_analysis/)
- [最终仿真目录](final_evaluation_model_50008/)
- [Nominal 回放摘要](final_evaluation_model_50008/nominal_seed42/analysis/summary.json)
- [随机 seed 0 回放摘要](final_evaluation_model_50008/robust_seed0/analysis/summary.json)
- [随机 seed 1 回放摘要](final_evaluation_model_50008/robust_seed1/analysis/summary.json)
- [随机 seed 2 回放摘要](final_evaluation_model_50008/robust_seed2/analysis/summary.json)
- [完整训练日志](training.log)

## 后续建议

1. 给 motion NPZ 补充 `joint_names`，把 joint 数组也改为显式名称映射。
2. 将 dirty diff、未跟踪脚本和 motion 数据打包成可复现 release。
3. 若洁净回放通过而随机化回放失败，按失败因素扩充定向评估，再决定继续训练或调整 curriculum。
4. 若希望机器人始终更靠平台几何中心，需先决定是否允许移除 reference root 的 0.487 m 水平位移；这与修复初始位置不同。
5. 部署时固定 observation/action 顺序、normalizer、100 Hz 时序和动作 scale。
