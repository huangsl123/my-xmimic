# DexEVT dance1：4096 环境冒烟训练与仿真分析

日期：2026-07-28  
任务：`Tracking-Flat-DexEVT-Wo-State-Estimation-v0`  
运行目录：`2026-07-28_13-37-30_dance1_4096_smoke_fixed`

## 结论

本次 4096 环境、10 轮冒烟训练完整跑通，共采集并训练了 **983,040 个 transition**，未出现 OOM、NaN、Inf 或 Python traceback，checkpoint、TensorBoard、ONNX、图表和仿真视频均已成功保存。

冒烟测试证明训练链路已恢复，但 **10 轮模型没有收敛**，不能视为可用的完整动作策略。固定种子的正式仿真在第 121 步（1.21 s 仿真时间）因 `anchor_pos` 终止，只完成 3972 帧动作的 3.07%。这与训练曲线一致：旧的末端身体映射错误已经消失，当前主要问题已经转变为策略尚未学会稳定保持骨盆高度和动作跟踪。

最关键的历史错误是 motion NPZ 有 39 个身体，而 Isaac Sim 5.0 导入 URDF 后只保留 24 个 articulation body。旧代码直接用运行时数字索引读取 39-body 文件，导致例如运行时左脚踝索引 18 实际读到 `camera_body_front_link`，目标高度约 1.30 m，而真正脚踝目标约 0.03 m。旧训练中几乎每步触发的 `ee_body_pos` 因此是数据映射错误，不是策略能力问题。现在已改为按 `body_names` 映射，并对无名字且布局不一致的文件直接报错。

## 本次运行配置

| 项目 | 值 |
|---|---:|
| 并行环境 | 4096 |
| PPO 轮数 | 10 |
| 每环境每轮步数 | 24 |
| 总 transition | 983,040 |
| 随机种子 | 42 |
| 物理步长 | 0.005 s |
| 控制步长 / motion FPS | 0.01 s / 100 Hz |
| Motion 帧数 | 3972 |
| Motion 时长 | 39.72 s |
| GPU | NVIDIA GeForce RTX 5090，32 GB |
| 端到端冒烟运行时间 | 约 26.3 s |
| 训练内循环时间 | 约 11.6 s |

训练使用带名字元数据的新 motion：
`../../motion_data/dance1_easy_named.npz`

## 训练结果

| 指标 | 第 0 轮 | 第 9 轮 | 解读 |
|---|---:|---:|---|
| Mean episode length | 16.75 | 89.31 | 已不再是历史上的每步重置，但仍只有约 0.89 s |
| Mean reward | -0.341 | -1.760 | 10 轮内未收敛；最低为 -2.559 |
| Anchor position error | 0.070 | 0.584 | 当前主要失稳来源 |
| Joint position error | 1.438 | 2.430 | 动作跟踪仍很弱 |
| `ee_body_pos` 终止统计 | 25.083 | 0.000 | 身体索引修复生效，第 1 轮后保持为 0 |
| `anchor_pos` 终止统计 | 0.000 | 46.292 | 真正的骨盆高度/位置失稳被暴露出来 |
| Value loss | 0.0255 | 0.00725 | 数值有限且下降 |
| Mean noise std | 0.9959 | 0.9002 | 正常下降 |
| Throughput | 46,473 | 93,312 steps/s | 平均 88,635 steps/s，峰值 94,452 |

TensorBoard 中的终止数值是 RSL-RL 的聚合日志统计，不应解释成“4096 个环境中的原始计数”。

奖励分量方面，第 9 轮最大的负项是 `action_rate_l2=-0.654`，其次是 `joint_torque_l2=-0.108`。正向跟踪项虽然已经出现，但尚不足以抵消动作变化惩罚。由于只有 10 轮，目前不建议立刻改奖励权重；应先延长训练，观察该比例是否自然改善。

### 训练可视化

- [训练总览](training_analysis/training_overview.png)
- [奖励分量](training_analysis/reward_components.png)
- [跟踪误差](training_analysis/tracking_errors.png)
- [完整标量 CSV](training_analysis/scalars.csv)
- [机器可读训练摘要](training_analysis/summary.json)
- [完整训练日志](training.log)

## 正式仿真结果

正式回放使用 `model_9.pt`、固定种子 42，并关闭训练期随机事件、观测噪声和调试坐标轴。物理与策略仍按 100 Hz 运行，视频编码为 **25 FPS**，所以是严格的 **0.25 倍速**，不会因为 `sleep` 改变仿真物理。

| 项目 | 结果 |
|---|---:|
| 仿真步数 | 122 |
| 仿真时长 | 1.22 s |
| 视频时长 | 4.88 s |
| 首次终止 | step 121 / 1.21 s |
| 终止项 | `anchor_pos` |
| `anchor_ori` / `ee_body_pos` | 均未触发 |
| 动作完成率 | 3.07% |
| 终止前 anchor 3D 位置误差 | 0.623 m |
| 终止前平均 body 位置误差 | 0.338 m |
| 终止前 joint 位置误差范数 | 1.099 |

`anchor_pos` 在本任务中调用的是 `bad_anchor_pos_z_only`，阈值为 0.25 m，因此这次终止明确表示骨盆高度偏差越界。CSV 终止行中的普通误差值已经是 Isaac Lab 自动重置后的值，分析图和摘要因此有意使用终止前一帧作为最后稳定样本。

### 速度与可读性

旧回放同时存在参数化节流和硬编码 `time.sleep(0.2)`，而视频本身没有使用对应的低 FPS，导致交互窗口和保存视频的速度语义不一致。现在：

- 交互窗口按 `step_dt / playback_speed` 做墙钟节流；
- headless 视频不拖慢物理，仅按 `100 × playback_speed` 设置编码 FPS；
- `--playback_speed 0.25` 生成 25 FPS 视频；
- 调试 marker 默认隐藏，需要时可用 `--show_debug_markers` 打开；
- 回放在首次真实 `done` 后停止，不再把多个自动重置回合拼成一个视频。

### 平台中心

旧 viewer 使用 `origin_type="world"`，但 5×10 个 8 m 地形 tile 的环境原点可以离世界原点几十米，因此相机看向世界原点时，机器人看起来没有位于当前平台中心。现在 viewer 使用 `origin_type="env"` 和 `env_index=0`。

实测位置进一步确认：

- 起点相对平台中心：`(0.0006, -0.00008) m`，可以视为正中心；
- 终止前：`(0.530, 0.219) m`；
- 离中心最大距离：`0.574 m`；
- 8 m tile 半宽为 4 m，该距离只占半宽的 **14.35%**。

所以初始物理位置已经正确；后续偏移是未收敛策略产生的真实漂移，而不是平台原点配置错误。

### 仿真产物

- [0.25 倍速正式仿真视频](simulation_final/rl-video-step-0.mp4)
- [关键帧总览](simulation_final/contact_sheet.png)
- [仿真误差与平台位置图](simulation_final/analysis/playback_analysis.png)
- [逐步指标 CSV](simulation_final/metrics.csv)
- [机器可读仿真摘要](simulation_final/analysis/summary.json)
- [完整回放日志](simulation_final/playback.log)

## ONNX 与部署检查

新的 ONNX 已包含经验归一化参数：

- 输入：`obs [1,124]`、`time_step [1,1]`
- 输出：actions 加 6 组 motion reference
- initializer：10 个
- 包含 `normalizer._mean [1,124]` 和归一化除数 `[1,124]`
- 包含 joint names、stiffness、damping、default pose、observation names 和 action scale 元数据

旧导出代码从不存在的 `policy.actor_obs_normalizer` 取归一化器，因此旧 ONNX 只有 actor 权重。现在训练 runner 和 play runner 都从 `runner.obs_normalizer` 导出。

- 训练保存的 ONNX：`../../logs/rsl_rl/dex_evt_flat/2026-07-28_13-37-30_dance1_4096_smoke_fixed/2026-07-28_13-37-30_dance1_4096_smoke_fixed.onnx`
- 回放重新导出的 ONNX：`../../logs/rsl_rl/dex_evt_flat/2026-07-28_13-37-30_dance1_4096_smoke_fixed/exported/policy.onnx`

## 建议的下一阶段

1. 使用本次修复后的 named motion 和 checkpoint 体系继续至少 1000 轮训练，再做固定种子评估。按本次稳定阶段约 1.05 s/轮估算，1000 轮训练内循环约 17.5 分钟，实际时间还会包含启动和保存开销。
2. 首要验收标准应是：固定种子回放不触发 `anchor_pos` / `ee_body_pos`，并能完成全部 3972 帧；其次才看 reward 的绝对值。
3. 如果延长训练后 `action_rate_l2` 仍长期压过全部正向奖励，再考虑奖励尺度、动作平滑权重或分阶段 curriculum；不要根据 10 轮冒烟结果直接调参。
4. 历史上使用无 `body_names` 的 39-body motion 训练出的模型和曲线应视为无效基线，不应与本次结果直接比较。

## 本次代码修复

- Motion loader 改为按身体名称映射，并验证缺失、重复、越界和 FPS。
- 提供 legacy 39-body motion 标注脚本。
- 修复全动作回放起始状态与 reference alignment。
- 修复回放速度、视频 FPS、自动退出、固定种子、终止原因和逐步指标记录。
- 相机从 world origin 改为 environment 0 origin。
- 回放默认关闭随机化、观测噪声和遮挡动作的调试 marker。
- 修复 ONNX observation normalizer 与无随机化模式下默认关节元数据导出。
- 增加训练与仿真分析脚本，输出 CSV、JSON 和 PNG。


