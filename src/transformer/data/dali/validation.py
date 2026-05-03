from __future__ import annotations

import os
import tempfile
import typing as t
from pathlib import Path

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def


@pipeline_def
def _dali_batch_validation_pipeline(file_list: str):
    encoded, _ = fn.readers.file(
        file_list=file_list,
        random_shuffle=False,
        name="Reader",
    )

    decoded = fn.decoders.image(
        encoded,
        device="mixed",
        output_type=types.RGB,
        hw_decoder_load=0.9,
    )

    return decoded


@pipeline_def
def _dali_single_image_validation_pipeline():
    encoded = fn.external_source(
        name="encoded",
        device="cpu",
        dtype=types.UINT8,
    )

    decoded = fn.decoders.image(
        encoded,
        device="mixed",
        output_type=types.RGB,
        hw_decoder_load=0.9,
    )

    return decoded


class DaliImageValidator:
    DEFAULT_SUPPORTED_FORMATS = {
        "JPEG", "PNG", "BMP", "TIFF", "PPM", "PGM", "PBM", "WEBP", "JPEG2000",
    }

    _SUFFIX_TO_FORMAT = {
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".png": "PNG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".ppm": "PPM",
        ".pgm": "PGM",
        ".pbm": "PBM",
        ".webp": "WEBP",
        ".jp2": "JPEG2000",
        ".j2k": "JPEG2000",
        ".j2c": "JPEG2000",
        ".jpc": "JPEG2000",
    }

    def __init__(
        self,
        supported_formats: t.Optional[t.Iterable[str]] = None,
        num_threads: t.Optional[int] = None,
        device_id: t.Optional[int] = None,
        delete_invalid: bool = True,
        batch_size: int = 256,
        progress_callback: t.Optional[t.Callable[[int], None]] = None,
    ) -> None:
        self.supported_formats = set(supported_formats or self.DEFAULT_SUPPORTED_FORMATS)
        self.num_threads = num_threads or max(4, os.cpu_count() or 4)
        self.device_id = 0 if device_id is None else int(device_id)
        self.delete_invalid = delete_invalid
        self.batch_size = max(1, int(batch_size))
        self.progress_callback = progress_callback
        self._single_image_pipe = None

    def _guess_format_from_path(self, image_path: Path) -> str:
        return self._SUFFIX_TO_FORMAT.get(image_path.suffix.lower(), "UNKNOWN")

    def _delete_file(self, image_path: Path) -> str:
        if not self.delete_invalid:
            return ""

        try:
            if image_path.exists() and image_path.is_file():
                image_path.unlink()
                return " [DELETED]"
        except Exception as exc:
            return f" [DELETE FAILED: {exc}]"

        return ""

    def _detect_unsupported_signature(self, image_path: Path) -> t.Optional[str]:
        try:
            with image_path.open("rb") as file:
                header = file.read(12)
        except Exception:
            return None

        if header.startswith((b"GIF87a", b"GIF89a")):
            return "GIF"

        return None

    def _summarize_exception(self, exc: Exception) -> str:
        text = str(exc)

        if "GIF images are not supported" in text:
            return "GIF images are not supported"

        for line in (line.strip() for line in text.splitlines() if line.strip()):
            if line in {
                "Critical error in pipeline:",
                "Current pipeline object is no longer valid.",
                ". File:",
            }:
                continue
            if line.startswith("Error when executing "):
                continue
            if line.startswith("Stacktrace") or line.startswith("[frame"):
                continue
            if "] " in line:
                line = line.rsplit("] ", 1)[-1]
            return line

        return exc.__class__.__name__

    def _precheck_path(self, image_path: Path) -> t.Optional[t.Tuple[bool, str]]:
        try:
            stat = image_path.stat()
        except FileNotFoundError:
            return False, "Invalid or corrupted image: file does not exist"

        if not image_path.is_file():
            return False, "Invalid or corrupted image: file does not exist"

        if stat.st_size == 0:
            delete_msg = self._delete_file(image_path)
            return False, f"Invalid or corrupted image: empty file{delete_msg}"

        image_format = self._guess_format_from_path(image_path)
        if image_format not in self.supported_formats:
            delete_msg = self._delete_file(image_path)
            return False, f"Unsupported DALI format: {image_format}{delete_msg}"

        disguised_format = self._detect_unsupported_signature(image_path)
        if disguised_format is not None:
            delete_msg = self._delete_file(image_path)
            return (
                False,
                f"Unsupported DALI format: {disguised_format} disguised as "
                f"{image_path.suffix or 'unknown'}{delete_msg}",
            )

        return None

    def _get_single_image_pipe(self):
        if self._single_image_pipe is None:
            self._single_image_pipe = _dali_single_image_validation_pipeline(
                batch_size=1,
                num_threads=self.num_threads,
                device_id=self.device_id,
                exec_pipelined=False,
                exec_async=False,
                prefetch_queue_depth=1,
            )
            self._single_image_pipe.build()

        return self._single_image_pipe

    def _drop_single_image_pipe(self) -> None:
        pipe = self._single_image_pipe
        self._single_image_pipe = None

        if pipe is not None:
            try:
                pipe.reset()
            except Exception:
                pass

    def _write_file_list(self, image_paths: t.Sequence[Path]) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix="_dali_validation.txt",
            delete=False,
            encoding="utf-8",
        ) as file:
            for index, image_path in enumerate(image_paths):
                file.write(f"{image_path} {index}\n")
            return file.name

    def _run_batch_pipeline_for_paths(self, image_paths: t.Sequence[Path]) -> None:
        if not image_paths:
            return

        file_list_path = self._write_file_list(image_paths)
        pipe = None

        try:
            pipe = _dali_batch_validation_pipeline(
                batch_size=len(image_paths),
                num_threads=self.num_threads,
                device_id=self.device_id,
                file_list=file_list_path,
                exec_pipelined=True,
                exec_async=True,
                prefetch_queue_depth=2,
            )
            pipe.build()
            outputs = pipe.run()
            del outputs
        finally:
            if pipe is not None:
                try:
                    pipe.reset()
                except Exception:
                    pass

            try:
                Path(file_list_path).unlink()
            except FileNotFoundError:
                pass

    def _run_single_pipeline_for_path(self, image_path: Path) -> None:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        if encoded.size == 0:
            raise ValueError("empty file")

        pipe = self._get_single_image_pipe()

        try:
            pipe.feed_input("encoded", [encoded])
            outputs = pipe.run()
            del outputs
        except Exception:
            self._drop_single_image_pipe()
            raise

    def _validate_path_with_dali(self, image_path: Path) -> t.Tuple[bool, str]:
        precheck_result = self._precheck_path(image_path)
        if precheck_result is not None:
            return precheck_result

        image_format = self._guess_format_from_path(image_path)

        try:
            self._run_single_pipeline_for_path(image_path)
            return True, image_format
        except Exception as exc:
            delete_msg = self._delete_file(image_path)
            reason = self._summarize_exception(exc)
            return False, f"Invalid or corrupted image: {reason}{delete_msg}"

    def _validate_chunk(
        self,
        image_paths: t.Sequence[Path],
    ) -> t.Iterator[t.Tuple[str, t.Tuple[bool, str]]]:
        results: t.List[t.Optional[t.Tuple[bool, str]]] = [None] * len(image_paths)
        decode_candidates: t.List[t.Tuple[int, Path, str]] = []

        for index, image_path in enumerate(image_paths):
            precheck_result = self._precheck_path(image_path)
            if precheck_result is not None:
                results[index] = precheck_result
            else:
                decode_candidates.append(
                    (index, image_path, self._guess_format_from_path(image_path))
                )

        if decode_candidates:
            candidate_paths = [path for _, path, _ in decode_candidates]

            try:
                self._run_batch_pipeline_for_paths(candidate_paths)
            except Exception:
                for index, image_path, _ in decode_candidates:
                    results[index] = self._validate_path_with_dali(image_path)
            else:
                for index, _, image_format in decode_candidates:
                    results[index] = True, image_format

        for image_path, result in zip(image_paths, results):
            if self.progress_callback is not None:
                self.progress_callback(1)

            if result is None:
                raise RuntimeError(f"Validation result missing for {image_path}")

            yield str(image_path), result

    def validate(self, path: t.Union[str, Path]) -> t.Tuple[bool, str]:
        return self._validate_path_with_dali(Path(path))

    def validate_many(
        self,
        paths: t.Iterable[t.Union[str, Path]],
    ) -> t.Iterator[t.Tuple[str, t.Tuple[bool, str]]]:
        chunk: t.List[Path] = []

        for raw_path in paths:
            chunk.append(Path(raw_path))

            if len(chunk) >= self.batch_size:
                yield from self._validate_chunk(chunk)
                chunk = []

        if chunk:
            yield from self._validate_chunk(chunk)
