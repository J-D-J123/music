"""Run transcription across multiple audio stems concurrently."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

from pipeline.transcribe import transcribe

MAX_WORKERS = min(3, os.cpu_count() or 1)

# TensorFlow initializes internal threads/locks that don't survive fork() safely —
# if any transcribe() call already ran in this process (e.g. for double-stop
# detection) before this pool starts, forked workers can inherit that state and
# hang. spawn avoids this by starting clean processes instead of cloning memory.
_ctx = multiprocessing.get_context("spawn")


def transcribe_many(stem_paths: dict, output_dir=None) -> dict:
    """stem_paths: {name: path}. Returns {name: midi_path}."""
    results = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=_ctx) as executor:
        futures = {
            executor.submit(transcribe, str(path), output_dir): name
            for name, path in stem_paths.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"[parallel_transcribe] done: {name}")
            except Exception as e:
                print(f"[parallel_transcribe] FAILED {name}: {e}")
    return results