# Pipette Right-Only Action DiT

This experiment is independent of the existing G1 real-robot pipelines. It uses only ego-view video,
the right arm/right dexterous hand state, and one of two right-only action spaces:

- `pipette_right_joints`: 7 right-arm joint commands + 6 right-hand commands.
- `pipette_right_wrist_delta`: wrist-frame relative translation (3) + relative rotation 6D (6) + 6 right-hand commands. Both state and action wrist poses are relative transforms; state is `T[t-1]^-1 T[t]`, while action is `T[t]^-1 T[t+1]`.

The Cosmos video backbone is frozen completely and the Action DiT is initialized randomly. No left-arm,
lower-body, waist, root-state, motion-token, SMPL, or planner fields are included in the derived parquet files.
Both training configs save every 4000 steps and retain only the newest two complete checkpoints.

Prepare datasets:

```bash
python examples/PipetteRightOnly/prepare_datasets.py \
  --source /home/nvme02/DiT4DiT_data/raw/pick_up_pipette \
  --output-root /home/nvme02/DiT4DiT_data/derived
```

Launch both two-GPU runs:

```bash
bash examples/PipetteRightOnly/launch_two_trainings.sh
```

On A800_1, the persistent project, datasets, model weights, and runs are under `/workspace/WM`.
The recoverable Conda snapshot is under `/workspace/conda_envs/dit4dit/current`, while the live
environment uses `/dev/shm/conda_envs/dit4dit` according to the server policy. Restore and validate
the environment before launching:

```bash
bash examples/PipetteRightOnly/restore_a800_environment.sh
bash examples/PipetteRightOnly/launch_two_trainings_a800.sh
```

The A800 launcher defaults to GPU pairs `0,1` and `2,3`. Override them after checking current GPU
occupancy, for example `JOINT_GPUS=4,5 WRIST_GPUS=6,7 bash .../launch_two_trainings_a800.sh`.
