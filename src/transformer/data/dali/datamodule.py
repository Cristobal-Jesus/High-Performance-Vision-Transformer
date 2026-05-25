"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristobal Jesus Sarmiento Rodriguez
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
import typing as t
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
        root_dir: t.Union[str, Path],
        train_batch_size: int = 128,
        val_batch_size: int = 128,
        image_size: int = 224,
        num_threads: int = 24,
        device_id: int = 0,
        seed: int = 42,
        train_ratio: float = 0.8,
        drop_last: bool = True,
        train_prefetch_queue_depth: int = 3,
        val_prefetch_queue_depth: int = 2,
        verbose_invalid: bool = True,
        delete_invalid_files: bool = True,
        pipeline_factory: t.Optional[DaliPipelineFactory] = None,
        validator: t.Optional[DaliImageValidator] = None,
        shard_id: int = 0,
        num_shards: int = 1,
        show_progress: bool = True,
        validator_batch_size: int = 128,
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
        self.delete_invalid_files = delete_invalid_files
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.show_progress = show_progress

        self.validator = validator or DaliImageValidator(
            num_threads=self.num_threads,
            device_id=self.device_id,
            delete_invalid=self.delete_invalid_files,
            batch_size=self.validator_batch_size,
        )
        self.split_manager = DatasetSplitManager(
            root_dir=self.root_dir,
            validator=self.validator,
            train_ratio=self.train_ratio,
            seed=self.seed,
            show_progress=self.show_progress,
        )
        self.pipeline_factory = pipeline_factory or DaliPipelineFactory()

        self.train_file_list = None  # type: t.Optional[str]
        self.val_file_list = None  # type: t.Optional[str]
        self.train_loader = None  # type: t.Optional[DALIGenericIterator]
        self.val_loader = None  # type: t.Optional[DALIGenericIterator]
        self.label_map = {}  # type: t.Dict[str, int]
        self._split_done = False

    def _ensure_split(self) -> None:
        if self._split_done:
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
        self._split_done = True

    def setup(self, stage: t.Optional[str] = None) -> None:
        """Prepare the DALI pipelines and iterators for training and validation."""
        if stage not in (None, "fit", "validate"):
            return

        build_train = stage in (None, "fit") and self.train_loader is None
        build_val = stage in (None, "fit", "validate") and self.val_loader is None

        if not (build_train or build_val):
            return

        self._ensure_split()

        if build_train:
            self.train_loader = self._build_train_loader()
        if build_val:
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
            raise RuntimeError(
                "The training loader has not been initialized. Call setup() first."
            )

        return self.train_loader

    def val_dataloader(self) -> DALIGenericIterator:
        """Return the validation data loader."""
        if self.val_loader is None:
            raise RuntimeError(
                "The validation loader has not been initialized. Call setup() first."
            )

        return self.val_loader

    def teardown(self, stage: t.Optional[str] = None) -> None:
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
        self._split_done = False
