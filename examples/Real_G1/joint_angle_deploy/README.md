# G1 右臂关节角实机部署

架构：A800_1 运行 DiT4DiT ZMQ 服务；连接 G1 DDS 网络的控制机运行客户端。当前
`pipette_right_joints_action_dit` 模型输入/输出均为右臂 7 维与右 Inspire 手 6 维。客户端
发布右臂到 `rt/arm_sdk`，并发布 12 维双手消息到 `rt/inspire/cmd`；右手使用模型输出，
左手保持 episode 0 的张开初态。`inspire_modbus_hand.py` 负责将 DDS 命令桥接到 Modbus。

## 1. A800_1 推理服务

```bash
cd /workspace/WM/dit4dit/DiT4DiT
/dev/shm/conda_envs/dit4dit/bin/python deployment/model_server/server_policy_zmq.py \
  --ckpt_path /workspace/WM/dit4dit/DiT4DiT_runs/pipette_right_joints_action_dit/checkpoints/steps_48000_pytorch_model.pt \
  --port 5556 --use_bf16
```

## 2. 启动 Inspire DDS/Modbus bridge

在连接 G1 和 Inspire 手的控制机、SONICMJ 项目根目录运行：

```bash
python decoupled_wbc/scripts/inspire_modbus_hand.py --mode dds \
  --network enp7s0 --hand-task pick_up_pipette \
  --hand-task-config gear_sonic/config/data_collection/inspire_hand_tasks.json \
  --profile-timing
```

## 3. 先测试急停

让 G1 使用吊架/可靠支撑，操作者手保持在键盘空格上。程序只让一个腕关节以 0.05 rad
幅度、10 秒周期缓慢运动；启动后仍需按 Enter 才会下发。

```bash
cd examples/Real_G1/joint_angle_deploy
python test_estop_slow.py --network-interface enp7s0
```

按 Space 或 Q 后应立即停止轨迹并保持触发时的实测位置。测试通过后再运行模型客户端。

## 4. 模型客户端

先不带 `--arm` 做网络、相机、输出维度和限位检查；确认持续打印合理目标后再加
`--arm`，并在程序启动后按 Enter 二次解锁：

```bash
python g1_joint_client.py --server <A800_1可达IP> --network-interface enp7s0 --camera 0
python g1_joint_client.py --server <A800_1可达IP> --network-interface enp7s0 --camera 0 --arm
```

真机启动顺序：

1. 程序打印初始化目标关节角，按 Enter 才启用 `arm_sdk`。
2. 右臂从 LowState 实测角出发，以 minimum-jerk 曲线移动至 episode 0 的精确初态
   `[0.010702, -0.233477, -0.072876, -0.584854, 0.365135, 0.419927, -0.250482]`；
   Inspire 手同步设为 `[0.998, 1, 0.998, 0.998, 0.999, 0.984]`（1 为张开）。
3. 程序显示 `READY` 后会持续保持初始姿态，但不会查询模型；检查现场后按 `L` 才开始推理。
4. 初始化和推理期间按 Space/Q 都会锁存急停。

默认初始化至少 5 秒且峰值速度不超过 0.15 rad/s。可以显式指定其它训练匹配姿态：

```bash
python g1_joint_client.py ... --arm \
  --initial-duration 8 --initial-speed 0.1 \
  --initial-right-arm 0.010702 -0.233477 -0.072876 -0.584854 0.365135 0.419927 -0.250482 \
  --initial-right-hand 0.998 1 0.998 0.998 0.999 0.984
```

下发使用 Unitree 官方 `rt/arm_sdk` overlay，保留原有下肢控制器。安全逻辑包括：键盘
急停锁存、Ctrl-C、低状态 200 ms 看门狗、推理 5 s 超时、NaN/形状检查、URDF 硬限位、
0.25 rad/s 手臂速度限制、0.5/s 灵巧手归一化速度限制；任何异常都会切换为手臂实测
位置保持与灵巧手最后命令保持。
