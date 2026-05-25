"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristóbal Jesús Sarmiento Rodríguez
Date: 17th March 2026
File: datamodule.py

Description:
    This file defines the PyTorch Lightning data module that coordinates
    dataset validation, train/validation splitting, DALI pipeline
    construction, and data loader creation.

References:
    - https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    - https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

import logging
from pathlib import Path

import pytorch_lightning as pl
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from .pipelines import DaliPipelineFactory
from .split_utils import DatasetSplitManager
from .validation import DaliImageValidator

logger = logging.getLogger(__name__)


class DaliDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module backed by NVIDIA DALI."""

    def __init__(
        self,
        root_dir: str | Path,
        train_batch_size: int = 1536,
        val_batch_size: int = 512,
        image_size: int = 224,
        num_threads: int = 24,
        device_id: int = 0,
        seed: int = 42,
        train_ratio: float = 0.8,
        drop_last: bool = True,
        train_prefetch_queue_depth: int = 6,
        val_prefetch_queue_depth: int = 2,
        verbose_invalid: bool = True,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size
        self.num_threads = num_threads
        self.device_id = device_id
        self.seed = seed
        self.train_ratio = train_ratio
        self.drop_last = drop_last
        self.train_prefetch_queue_depth = train_prefetch_queue_depth
        self.val_prefetch_queue_depth = val_prefetch_queue_depth
        self.verbose_invalid = verbose_invalid

        self.validator = DaliImageValidator()
        self.split_manager = DatasetSplitManager(
            root_dir=self.root_dir,
            validator=self.validator,
            train_ratio=self.train_ratio,
            seed=self.seed,
        )
        self.pipeline_factory = DaliPipelineFactory()

        self.train_file_list: str | None = None
        self.val_file_list: str | None = None
        self.train_loader: DALIGenericIterator | None = None
        self.val_loader: DALIGenericIterator | None = None
        self.label_map: dict[str, int] = {}

    def setup(self, stage: str | None = None) -> None:
        """Prepare the DALI pipelines and iterators for training and validation."""
        if stage not in (None, "fit", "validate"):
            return

        if self.train_loader is not None and self.val_loader is not None:
            return

        x_train, y_train, x_val, y_val, self.label_map, invalid_files = (
            self.split_manager.create_split()
        )

        if invalid_files:
            logger.warning("Skipped %d invalid or unsupported files.", len(invalid_files))

            if self.verbose_invalid:
                for path, reason in invalid_files[:50]:
                    logger.warning("[DALI][SKIP] %s -> %s", path, reason)

                if len(invalid_files) > 50:
                    logger.warning(
                        "... and %d more invalid files.",
                        len(invalid_files) - 50,
                    )

        self.train_file_list = self.split_manager.write_file_list(
            x_train,
            y_train,
            "_train_list.txt",
        )
        self.val_file_list = self.split_manager.write_file_list(
            x_val,
            y_val,
            "_val_list.txt",
        )

        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_val_loader()

    def _build_train_loader(self) -> DALIGenericIterator:
        """Create the DALI iterator used during training."""
        if self.train_file_list is None:
            raise RuntimeError("Training file list has not been created.")

        train_pipe = self.pipeline_factory.create_train_pipeline(
            batch_size=self.train_batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            file_list=self.train_file_list,
            crop=self.image_size,
            prefetch_queue_depth=self.train_prefetch_queue_depth,
        )
        train_pipe.build()

        last_batch_policy = (
            LastBatchPolicy.DROP if self.drop_last else LastBatchPolicy.PARTIAL
        )

        return DALIGenericIterator(
            [train_pipe],
            output_map=["images", "labels"],
            reader_name="Reader",
            last_batch_policy=last_batch_policy,
            auto_reset=True,
        )

    def _build_val_loader(self) -> DALIGenericIterator:
        """Create the DALI iterator used during validation."""
        if self.val_file_list is None:
            raise RuntimeError("Validation file list has not been created.")

        val_pipe = self.pipeline_factory.create_val_pipeline(
            batch_size=self.val_batch_size,
            num_threads=self.num_threads,
            device_id=self.device_id,
            file_list=self.val_file_list,
            size=self.image_size,
            prefetch_queue_depth=self.val_prefetch_queue_depth,
        )
        val_pipe.build()

        return DALIGenericIterator(
            [val_pipe],
            output_map=["images", "labels"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )

    def train_dataloader(self) -> DALIGenericIterator:
        """Return the training data loader."""
        if self.train_loader is None:
            raise RuntimeError("The training loader has not been initialized. Call setup() first.")

        return self.train_loader

    def val_dataloader(self) -> DALIGenericIterator:
        """Return the validation data loader."""
        if self.val_loader is None:
            raise RuntimeError("The validation loader has not been initialized. Call setup() first.")

        return self.val_loader

    def teardown(self, stage: str | None = None) -> None:
        """Remove temporary file lists created for DALI."""
        for file_path in (self.train_file_list, self.val_file_list):
            if file_path is None:
                continue

            path = Path(file_path)
            if path.exists():
                path.unlink()

        self.train_file_list = None
        self.val_file_list = None
        self.train_loader = None
        self.val_loader = None
