# Pipette Right-Only Action DiT

This experiment is independent of the existing G1 real-robot pipelines. It uses only ego-view video,
the right arm/right dexterous hand state, and one of two right-only action spaces:

- `pipette_right_joints`: 7 right-arm joint commands + 6 right-hand commands.
- `pipette_right_wrist_delta`: wrist-frame relative translation (3) + relative rotation 6D (6) + 6 right-hand commands.

The Cosmos video backbone is frozen completely and the Action DiT is initialized randomly. No left-arm,
lower-body, waist, root-state, motion-token, SMPL, or planner fields are included in the derived parquet files.

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
