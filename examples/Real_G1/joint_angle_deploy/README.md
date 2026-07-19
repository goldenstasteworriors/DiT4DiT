# G1 右臂关节角实机部署

架构：A800_1 运行 DiT4DiT ZMQ 服务；连接 G1 DDS 网络的控制机运行客户端。当前
`pipette_right_joints_action_dit` 模型输入为右臂 7 维状态，输出为右臂 7 维关节角与
右 Inspire 手 6 维（客户端当前只下发右臂，手部需使用现有 Inspire bridge）。

## 1. A800_1 推理服务

```bash
cd /workspace/WM/dit4dit/DiT4DiT
/dev/shm/conda_envs/DiT4DiT_env/bin/python deployment/model_server/server_policy_zmq.py \
  --ckpt_path /workspace/WM/dit4dit/DiT4DiT_runs/pipette_right_joints_action_dit/checkpoints/steps_48000_pytorch_model.pt \
  --port 5556 --use_bf16
```

## 2. 先测试急停

让 G1 使用吊架/可靠支撑，操作者手保持在键盘空格上。程序只让一个腕关节以 0.05 rad
幅度、10 秒周期缓慢运动；启动后仍需按 Enter 才会下发。

```bash
cd examples/Real_G1/joint_angle_deploy
python test_estop_slow.py --network-interface enp3s0
```

按 Space 或 Q 后应立即停止轨迹并保持触发时的实测位置。测试通过后再运行模型客户端。

## 3. 模型客户端

先不带 `--arm` 做网络、相机、输出维度和限位检查；确认持续打印合理目标后再加
`--arm`，并在程序启动后按 Enter 二次解锁：

```bash
python g1_joint_client.py --server <A800_1可达IP> --network-interface enp3s0 --camera 0
python g1_joint_client.py --server <A800_1可达IP> --network-interface enp3s0 --camera 0 --arm
```

下发使用 Unitree 官方 `rt/arm_sdk` overlay，保留原有下肢控制器。安全逻辑包括：键盘
急停锁存、Ctrl-C、低状态 200 ms 看门狗、推理 5 s 超时、NaN/形状检查、URDF 硬限位、
0.25 rad/s 速度限制；任何异常都会切换为实测位置保持。
