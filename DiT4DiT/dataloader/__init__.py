import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(
    cfg,
    dataset_py="lerobot_datasets_oxe",
    data_cfg=None,
    mode="train",
    normalization_metadata=None,
    save_statistics=True,
):

    if dataset_py == "lerobot_datasets":
        from DiT4DiT.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = data_cfg if data_cfg is not None else cfg.datasets.vla_data

        sharing_strategy = getattr(vla_dataset_cfg, "multiprocessing_sharing_strategy", None)
        if sharing_strategy:
            import torch.multiprocessing as mp

            mp.set_sharing_strategy(sharing_strategy)

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg, mode=mode)
        if normalization_metadata is not None:
            for dataset in vla_dataset.datasets:
                dataset.set_transforms_metadata(normalization_metadata[dataset.tag])
        num_workers = getattr(vla_dataset_cfg, "num_workers", 4)
        dataloader_kwargs = {}
        if num_workers > 0:
            dataloader_kwargs.update(
                persistent_workers=getattr(vla_dataset_cfg, "persistent_workers", True),
                prefetch_factor=getattr(vla_dataset_cfg, "prefetch_factor", 2),
            )

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            **dataloader_kwargs,
            # shuffle=True
        )        
        if save_statistics and dist.get_rank() == 0:
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
        
    else:
        raise NotImplementedError(f"Dataset {dataset_py} is not supported yet")
        
