"""
University of La Laguna
Higher School of Engineering and Technology
Bachelor's Degree in Computer Engineering
Bachelor's Thesis 2025-2026

Title: High-Computing Perfomance and Machine Learning
Author: Cristobal Jesus Sarmiento Rodriguez
Date: 17th March 2026
File: pipelines.py

Description:
    This file defines the class responsible for building the NVIDIA DALI
    pipelines used for training and validation image preprocessing.

References:
    - https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
"""

import typing as t

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import math as dali_math
from nvidia.dali.pipeline import pipeline_def


class DaliPipelineFactory:
    """Build NVIDIA DALI pipelines for training and validation."""

    def __init__(
        self,
        mean: t.Optional[t.Sequence[float]] = None,
        std: t.Optional[t.Sequence[float]] = None,
        horizontal_flip_probability: float = 0.5,
        grayscale_probability: float = 0.15,
        erasing_probability: float = 0.35,
        rotation_angle: float = 20.0,
        brightness_range: t.Tuple[float, float] = (0.75, 1.25),
        contrast_range: t.Tuple[float, float] = (0.75, 1.25),
        saturation_range: t.Tuple[float, float] = (0.70, 1.30),
        hue_range: t.Tuple[float, float] = (-20.0, 20.0),
        erasing_area_range: t.Tuple[float, float] = (0.02, 0.20),
        erasing_aspect_ratio_range: t.Tuple[float, float] = (0.3, 3.3),
        decoder_cache_size: int = 0,
        decoder_cache_threshold: int = 0,
        train_decoder_cache_size: int = 0,
        hw_decoder_load: float = 0.9,
        device_memory_padding: int = 211_025_920,
        host_memory_padding: int = 140_544_000,
        preallocate_width_hint: int = 1920,
        preallocate_height_hint: int = 1080,
        reader_prefetch_queue_depth: int = 4,
        output_dtype: types.DALIDataType = types.FLOAT16,
        read_ahead: bool = True,
    ) -> None:
        self.mean = list(mean or [0.485 * 255, 0.456 * 255, 0.406 * 255])
        self.std = list(std or [0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.horizontal_flip_probability = horizontal_flip_probability
        self.grayscale_probability = grayscale_probability
        self.erasing_probability = erasing_probability
        self.rotation_angle = rotation_angle

        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.erasing_area_range = erasing_area_range
        self.erasing_aspect_ratio_range = erasing_aspect_ratio_range

        self.decoder_cache_size = decoder_cache_size
        self.decoder_cache_threshold = decoder_cache_threshold
        self.train_decoder_cache_size = train_decoder_cache_size
        self.hw_decoder_load = hw_decoder_load
        self.device_memory_padding = device_memory_padding
        self.host_memory_padding = host_memory_padding
        self.preallocate_width_hint = preallocate_width_hint
        self.preallocate_height_hint = preallocate_height_hint
        self.reader_prefetch_queue_depth = reader_prefetch_queue_depth
        self.output_dtype = output_dtype
        self.read_ahead = read_ahead

    def create_train_pipeline(
        self,
        *,
        batch_size: int,
        num_threads: int,
        device_id: int,
        file_list: str,
        crop: int = 224,
        prefetch_queue_depth: int = 3,
    ):
        """Create the DALI pipeline used during training."""
        mean = self.mean
        std = self.std

        horizontal_flip_probability = self.horizontal_flip_probability
        grayscale_probability = self.grayscale_probability
        erasing_probability = self.erasing_probability
        rotation_angle = self.rotation_angle

        brightness_range = self.brightness_range
        contrast_range = self.contrast_range
        saturation_range = self.saturation_range
        hue_range = self.hue_range
        erasing_area_range = self.erasing_area_range
        erasing_aspect_ratio_range = self.erasing_aspect_ratio_range

        hw_decoder_load = self.hw_decoder_load
        device_memory_padding = self.device_memory_padding
        host_memory_padding = self.host_memory_padding
        preallocate_width_hint = self.preallocate_width_hint
        preallocate_height_hint = self.preallocate_height_hint
        reader_prefetch_queue_depth = self.reader_prefetch_queue_depth
        output_dtype = self.output_dtype
        read_ahead = self.read_ahead
        train_decoder_cache_size = self.train_decoder_cache_size
        decoder_cache_threshold = self.decoder_cache_threshold

        @pipeline_def(enable_conditionals=True)
        def pipeline(file_list: str, crop: int = 224):
            images, labels = fn.readers.file(
                file_list=file_list,
                random_shuffle=True,
                name="Reader",
                stick_to_shard=True,
                read_ahead=read_ahead,
                prefetch_queue_depth=reader_prefetch_queue_depth,
                skip_cached_images=train_decoder_cache_size > 0,
            )

            if train_decoder_cache_size > 0:
                # Caché activo: decodifica la imagen completa (cacheable) y
                # aplica el crop aleatorio como op separada en GPU.
                # A partir de la época 2 el disco queda inactivo.
                decoder_kwargs = dict(
                    device="mixed",
                    output_type=types.RGB,
                    hw_decoder_load=hw_decoder_load,
                    device_memory_padding=device_memory_padding,
                    host_memory_padding=host_memory_padding,
                    preallocate_width_hint=preallocate_width_hint,
                    preallocate_height_hint=preallocate_height_hint,
                    cache_size=train_decoder_cache_size,
                    cache_type="threshold",
                    cache_threshold=decoder_cache_threshold,
                )
                images = fn.decoders.image(images, **decoder_kwargs)
                images = fn.random_resized_crop(
                    images,
                    device="gpu",
                    size=[crop, crop],
                    random_area=[0.5, 1.0],
                    random_aspect_ratio=[0.67, 1.50],
                )
            else:
                # Sin caché: decoder fusionado con el crop (más eficiente
                # porque el JPEG solo decodifica los píxeles del crop).
                images = fn.decoders.image_random_crop(
                    images,
                    device="mixed",
                    output_type=types.RGB,
                    random_area=[0.5, 1.0],
                    random_aspect_ratio=[0.67, 1.50],
                    hw_decoder_load=hw_decoder_load,
                    device_memory_padding=device_memory_padding,
                    host_memory_padding=host_memory_padding,
                    preallocate_width_hint=preallocate_width_hint,
                    preallocate_height_hint=preallocate_height_hint,
                )
                images = fn.resize(
                    images,
                    device="gpu",
                    resize_x=crop,
                    resize_y=crop,
                )

            brightness = fn.random.uniform(range=brightness_range)
            contrast = fn.random.uniform(range=contrast_range)
            hue = fn.random.uniform(range=hue_range)

            saturation_value = fn.random.uniform(range=saturation_range)
            do_grayscale = fn.cast(
                fn.random.coin_flip(probability=grayscale_probability),
                dtype=types.FLOAT,
            )
            saturation = saturation_value * (1.0 - do_grayscale)

            images = fn.color_twist(
                images,
                device="gpu",
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )

            # Rotación aleatoria ±rotation_angle grados.
            # fill_value=0 rellena las esquinas con negro; keep_size=True
            # mantiene las dimensiones de salida iguales al crop original.
            angle = fn.random.uniform(range=(-rotation_angle, rotation_angle))
            images = fn.rotate(
                images,
                device="gpu",
                angle=angle,
                fill_value=0,
                keep_size=True,
            )

            do_erase = fn.random.coin_flip(probability=erasing_probability)
            if do_erase:
                erase_area = fn.random.uniform(range=erasing_area_range)
                erase_ratio = fn.random.uniform(range=erasing_aspect_ratio_range)
                erase_h = dali_math.sqrt(erase_area * erase_ratio)
                erase_w = dali_math.sqrt(erase_area / erase_ratio)
                erase_shape = fn.stack(erase_h, erase_w)

                anchor_y = fn.random.uniform(range=(0.0, 1.0)) * (1.0 - erase_h)
                anchor_x = fn.random.uniform(range=(0.0, 1.0)) * (1.0 - erase_w)
                erase_anchor = fn.stack(anchor_y, anchor_x)

                images = fn.erase(
                    images,
                    device="gpu",
                    anchor=erase_anchor,
                    shape=erase_shape,
                    axis_names="HW",
                    fill_value=0,
                    normalized_anchor=True,
                    normalized_shape=True,
                )

            mirror = fn.random.coin_flip(probability=horizontal_flip_probability)

            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=output_dtype,
                output_layout="CHW",
                mirror=mirror,
                mean=mean,
                std=std,
            )

            return images, labels

        return pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            file_list=file_list,
            crop=crop,
            prefetch_queue_depth=prefetch_queue_depth,
        )

    def create_val_pipeline(
        self,
        *,
        batch_size: int,
        num_threads: int,
        device_id: int,
        file_list: str,
        size: int = 224,
        prefetch_queue_depth: int = 2,
    ):
        """Create the DALI pipeline used during validation."""
        mean = self.mean
        std = self.std

        # La caché de imágenes DALI es un singleton por device: solo puede
        # inicializarse una vez.  Cuando el pipeline de entrenamiento ya la
        # reclamó (train_decoder_cache_size > 0), el pipeline de validación
        # no debe intentar usarla — DALI compara TODOS los parámetros internos
        # y lanza "already initialized with other parameters" si difieren.
        # La validación corre como máximo cada 5 épocas (~40 veces en 200
        # épocas), así que leer val desde disco no es un cuello de botella.
        decoder_cache_size = (
            0
            if self.train_decoder_cache_size > 0
            else self.decoder_cache_size
        )
        decoder_cache_threshold = self.decoder_cache_threshold
        hw_decoder_load = self.hw_decoder_load
        device_memory_padding = self.device_memory_padding
        host_memory_padding = self.host_memory_padding
        preallocate_width_hint = self.preallocate_width_hint
        preallocate_height_hint = self.preallocate_height_hint
        reader_prefetch_queue_depth = self.reader_prefetch_queue_depth
        output_dtype = self.output_dtype
        read_ahead = self.read_ahead

        @pipeline_def
        def pipeline(file_list: str, size: int = 224):
            images, labels = fn.readers.file(
                file_list=file_list,
                random_shuffle=False,
                name="Reader",
                stick_to_shard=True,
                read_ahead=read_ahead,
                prefetch_queue_depth=reader_prefetch_queue_depth,
                skip_cached_images=decoder_cache_size > 0,
            )

            decoder_kwargs = dict(
                device="mixed",
                output_type=types.RGB,
                hw_decoder_load=hw_decoder_load,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                preallocate_width_hint=preallocate_width_hint,
                preallocate_height_hint=preallocate_height_hint,
            )
            if decoder_cache_size > 0:
                decoder_kwargs["cache_size"] = decoder_cache_size
                decoder_kwargs["cache_type"] = "threshold"
                decoder_kwargs["cache_threshold"] = decoder_cache_threshold

            images = fn.decoders.image(images, **decoder_kwargs)

            images = fn.resize(
                images,
                device="gpu",
                resize_x=size,
                resize_y=size,
            )

            images = fn.crop_mirror_normalize(
                images,
                device="gpu",
                dtype=output_dtype,
                output_layout="CHW",
                mean=mean,
                std=std,
            )

            return images, labels

        return pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            file_list=file_list,
            size=size,
            prefetch_queue_depth=prefetch_queue_depth,
        )
