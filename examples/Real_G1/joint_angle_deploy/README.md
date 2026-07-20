# G1 右臂关节角实机部署

架构：A800_1 运行 DiT4DiT ZMQ 服务；连接 G1 DDS 网络的控制机运行客户端。当前
`pipette_right_joints_action_dit` 模型输入/输出均为右臂 7 维与右 Inspire 手 6 维。客户端
按 SONICMJ 的真机方式释放运动服务并发布 29 电机消息到 `rt/lowcmd`，同时发布 12 维
双手消息到 `rt/inspire/cmd`；右手使用模型输出，
左手保持 episode 0 的张开初态。`inspire_modbus_hand.py` 负责将 DDS 命令桥接到 Modbus。
客户端同时订阅 bridge 发布的 `rt/inspire/state`，READY 判定和模型状态输入使用手指实测值，
而不是上一次下发的命令值。

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

## 3. 启动机器人相机服务

在机器人 `192.168.123.164` 上启动采集时使用的 SONIC 相机服务：

```bash
cd <机器人上的GR00T-WholeBodyControl目录>
source .venv_camera/bin/activate
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera oak \
  --port 5555
```

如果机器人的 OAK 相机需要指定 MxID，再增加
`--ego-view-device-id <EGO_MXID>`。在控制机上用 `nc -vz 192.168.123.164 5555` 确认端口
可达。部署客户端默认读取训练数据使用的 `ego_view`；只有显式传入 `--camera-host ""`
时才会退回 `--camera 0` 指定的 PC 本地相机。

## 4. 先测试急停

必须让 G1 使用可靠吊架，操作者手保持在键盘空格上。按 Enter 后程序会释放宇树运动
服务并进入 `rt/lowcmd`；只让一个腕关节以 0.05 rad 幅度、10 秒周期缓慢运动。
双臂使用位置 PD，双腿和腰默认只有速度阻尼，不具有主动站立和平衡能力。

```bash
cd examples/Real_G1/joint_angle_deploy
python test_estop_slow.py --network-interface enp7s0
```

按 Space 或 Q 后应立即停止轨迹并保持触发时的实测位置。测试通过后再运行模型客户端。

## 5. 模型客户端

先不带 `--arm` 做网络、相机、输出维度和限位检查；确认持续打印合理目标后再加
`--arm`，并在程序启动后按 Enter 二次解锁：

```bash
python g1_joint_client.py --server <A800_1可达IP> --network-interface enp7s0 --view-camera
python g1_joint_client.py --server <A800_1可达IP> --network-interface enp7s0 --view-camera --arm
```

真机启动顺序：

1. 程序打印初始化目标关节角，按 Enter 才释放运动服务并启用 `rt/lowcmd`。
2. 双臂从 LowState 实测角出发，以同一条 minimum-jerk 曲线移动至 episode 0 的
   `timestamp=3.0 s` 姿态（parquet 第 150 行）。左臂为
   `[0.133169, 0.162950, 0.432475, -0.277567, -0.154381, 0.039344, -0.230636]`；
   右臂为
   `[-0.361888, -0.192083, 0.336661, -0.459164, 0.393083, 0.593854, -0.440780]`；
   双手平滑下发 episode 0 当时的 `action.wbc=[1,1,1,1,1,1]`。程序通过
   `rt/inspire/state` 检查 episode 0 实测状态：左手
   `[0.999,0.998,0.998,0.998,0.999,0.983]`，右手
   `[0.998,1,0.998,0.998,0.999,0.984]`。
   LowCmd 插值终点直接使用 episode 0 的 `observation.state`。插值结束后根据 LowState
   实测误差缓慢累积一个限速的位置修正量。默认修正速度不超过 `0.03 rad/s`，每个关节
   最多修正 `0.15 rad`；误差小于 `0.003 rad` 时停止积分，避免测量噪声导致漂移。
3. 程序会同时检查双臂与双手实测误差；手臂全部小于 `0.01 rad`、手指全部小于 `0.02`，
   并连续保持 1 秒才显示 `READY`。READY 后持续保持初始姿态，但不会查询模型；检查现场
   后按 `L` 才开始推理。
4. 初始化和推理期间按 Space/Q 都会锁存急停。

添加 `--view-camera` 后，dry-run 会立即打开机器人 `ego_view`；正式部署则在初始化完成、
进入 `READY` 后自动打开。相机采集与显示在独立线程中，A800 推理期间画面仍持续刷新。
窗口获得焦点时可按 `L` 开始推理，按 Space/Q 急停；终端快捷键同时有效。
客户端按 `ego_view` 帧时间戳判断画面是否真正更新；默认连续 `0.5 s` 没有新帧才打印
`[CAMERA WARNING]`，且每 2 秒最多打印一次。SONIC 客户端原先较敏感的 100 ms
stale 提示已关闭，可用 `--camera-stale-warning` 调整部署告警阈值。
如果 `cv2.getBuildInformation()` 显示 `GUI: NONE`，说明 `opencv-python-headless` 覆盖了
GUI 绑定；需在 `decoupled_vla_collection` 环境中安装与当前 OpenCV 同版本的
`opencv-python`。相机/显示线程异常会回传主循环，并进入既有的安全保持流程。

默认初始化至少 5 秒且峰值速度不超过 0.15 rad/s。可以显式指定其它训练匹配姿态：

```bash
python g1_joint_client.py ... --arm \
  --initial-duration 8 --initial-speed 0.1 \
  --initial-left-arm 0.133169 0.162950 0.432475 -0.277567 -0.154381 0.039344 -0.230636 \
  --initial-right-arm -0.361888 -0.192083 0.336661 -0.459164 0.393083 0.593854 -0.440780 \
  --initial-correction-rate 1.0 --initial-correction-speed 0.03 \
  --initial-correction-limit 0.15 --initial-correction-deadband 0.003 \
  --initial-left-hand-state 0.999 0.998 0.998 0.998 0.999 0.983 \
  --initial-right-hand-state 0.998 1 0.998 0.998 0.999 0.984 \
  --initial-hand-command 1 1 1 1 1 1
```

下发参考 SONICMJ 的 `rt/lowcmd` 低层控制。初始化时双臂同步移动到 episode 0；推理时
右臂跟随模型，左臂保持 episode 0 姿态。双腿和腰默认使用纯速度阻尼（无位置目标、
零前馈力矩），因此必须可靠吊挂。
只有在确认机器人完全由吊架承重后，才可显式使用 `--lower-body-mode zero-torque`。
安全逻辑包括：键盘
急停锁存、Ctrl-C、低状态 200 ms 看门狗、推理 15 s 超时（包含首次 CUDA warm-up）、
NaN/形状检查、URDF 硬限位、
0.25 rad/s 手臂速度限制、0.5/s 灵巧手归一化速度限制；任何异常都会切换为手臂实测
位置保持与灵巧手最后命令保持。独立的 100 Hz 发布线程会在 A800 推理期间持续发布
最后一个安全 `rt/lowcmd`，避免远程推理造成低层命令断流。

## 6. 播放已保存的右臂轨迹

`play_right_arm_trajectory.py` 读取完整 episode 推理 NPZ，并从原始采集 Parquet 首帧读取
对应 episode 的左右臂初始姿态；初始化阶段双臂同步移动，播放阶段左臂保持该初始姿态、
右臂播放推理轨迹。脚本不发布
`rt/inspire/cmd`。因此灵巧手可以由单独的 DDS/Modbus bridge 运行用户指定的任务：

```bash
python decoupled_wbc/scripts/inspire_modbus_hand.py --mode dds \
  --network enp7s0 --hand-task grab_pipette \
  --hand-task-config gear_sonic/config/data_collection/inspire_hand_tasks.json \
  --profile-timing
```

先离线检查轨迹的形状、有限值、URDF 限位和相邻目标速度；此命令不连接 DDS：

```bash
conda run --no-capture-output -n decoupled_vla_collection python \
  examples/Real_G1/joint_angle_deploy/play_right_arm_trajectory.py
```

确认 G1 已由可靠吊架完全承重、灵巧手 bridge 正常后，再启用真机下发：

```bash
conda run --no-capture-output -n decoupled_vla_collection python \
  examples/Real_G1/joint_angle_deploy/play_right_arm_trajectory.py \
  --network-interface enp7s0 --lower-body-mode damping \
  --initial-duration 8 --initial-speed 0.1 --arm
```

脚本默认通过 `--episode 0` 在 `inference_records/` 中自动选择该 episode 的最高 steps
完整推理 NPZ。它对每帧的
16 步预测做因果重叠窗口平均，以2倍慢放（`--slowdown 2`）推进607个目标，并在离线阶段
生成最终100 Hz命令序列。每周期变化被硬限制为0.25 rad/s，随后再次检查 NaN/Inf、URDF
限位和最终命令峰值；任何检查失败都会在连接 DDS 前退出。先以 minimum-jerk 插值将
左右臂同步移动到对应 episode 的首帧输入姿态，到达 `READY` 后必须按 `L` 才连续播放完整轨迹。
minimum-jerk 插值结束后，初始化使用与在线部署相同的双臂外环误差修正：默认积分率
`1.0/s`、修正速度上限 `0.03 rad/s`、修正偏置上限 `0.15 rad`、死区 `0.003 rad`，
所有关节误差需连续1秒保持在 `0.01 rad` 内才进入 `READY`。内环使用
`g1_joint_client.py` 中相同的手臂 Kp/Kd。
播放完成后保持末姿态。
任意阶段按 Space/Q 或 Ctrl-C 都会锁存急停，
短暂保持触发时的实测角后停止发布。腿和腰默认只使用速度阻尼，不提供主动平衡能力。

真机前可以用 MuJoCo 播放完全相同的聚合、2倍慢放和限速后命令序列：

```bash
python examples/Real_G1/joint_angle_deploy/render_full_trajectory_mujoco.py
```

选择其它推理 episode 时，仿真和真机使用相同的参数：

```bash
# 仿真 episode 3；自动选择 episode 3 的最高 steps 结果
python examples/Real_G1/joint_angle_deploy/render_full_trajectory_mujoco.py --episode 3 --viewer

# 真机 episode 3
python examples/Real_G1/joint_angle_deploy/play_right_arm_trajectory.py --episode 3 \
  --network-interface enp7s0 --arm
```

如需指定某个旧权重的结果，可在两个脚本中都使用
`--trajectory inference_records/joints_steps_XXXXX_episode_000003_full.npz`，它会覆盖
`--episode` 的自动查找。

默认生成 `inference_records/g1_full_episode0_2x_slow_simulation.mp4`。如需交互窗口：

```bash
python examples/Real_G1/joint_angle_deploy/render_full_trajectory_mujoco.py --viewer
```

采集数据的旧41维格式可直接回放实测 `observation.state`：

```bash
/home/ykj/project/SONICMJ/GR00T-WholeBodyControl/.venv/bin/python \
  examples/Real_G1/joint_angle_deploy/render_collected_episode_mujoco.py \
  --dataset /home/ykj/project/SONICMJ/GR00T-WholeBodyControl/outputs/pick_up_pipette \
  --episode 0
```

添加 `--source action` 可改为回放当时的 `action.wbc`，添加 `--viewer` 可使用交互窗口。
该脚本会严格检查旧格式 `modality.json` 的41维布局，不会猜测或自动兼容后续新格式。
`--viewer` 使用 GLFW/X11 窗口后端；即使命令行误设了 `MUJOCO_GL=egl` 或 `osmesa`，
脚本也会在导入 MuJoCo 前自动忽略该离屏设置。导出 MP4 时仍可使用 `MUJOCO_GL=egl`。
