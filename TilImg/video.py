import os
import subprocess
from pathlib import Path
from typing import Union

from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn, \
    SpinnerColumn
from rich.table import Table

import TilImg
import colorama
import numpy
import numpy as np
import cv2
import rich
import tqdm
import typer
from PIL import Image

from TilImg import ClockTimer, generate_tiledimage_gu, utils
from TilImg.utils import ProcessType


class Video:
    def __init__(self, path: Path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path.absolute()))
        if not self.cap.isOpened():
            raise IOError(f"Couldn't video file at {path.absolute()}")


    @property
    def TotalFramesCount(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def Framerate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self)->float:
        return self.cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)
    @property
    def height(self)->float:
        return self.cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)

    def getFrame(self) -> np.ndarray:
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            yield frame

    def __del__(self):
        self.cap.release()


def generate_tiledimage_video(reference_video: Video, tiles: np.ndarray, tile_shape: tuple[int], useCuda=False,
                              resize_factor: float = -1, progress: Union[Progress, None] = None):
    if resize_factor < 0:
        if resize_factor == -1:
            resize_factor = 1 / max(tile_shape)
        else:
            typer.echo(f"Invalid resize_factor: {resize_factor}")
            return

    if progress:
        overallProcessTask = progress.add_task(f"Processing {reference_video.path.name} Frames",
                                               total=reference_video.TotalFramesCount)

        progress.print(f"Generating tiled image from video with {reference_video.path}")
        progress.print(f"Video frames count: {reference_video.TotalFramesCount}")
        progress.print(f"Video framerate count: {reference_video.Framerate}")
        progress.print(f"Will resize frames by a factor of {resize_factor}")
        progress.print(f"Tiles shape: {tiles.shape}")
        progress.print(f"useCuda: {useCuda}")

    os.makedirs(Path("./build/frames"), exist_ok=True)

    timer = ClockTimer()
    timer.start()
    index = 0

    for frame in reference_video.getFrame():

        save_filepath = Path(f"./build/frames/{reference_video.path.name}_{index}.png")
        if save_filepath.exists():
            if progress:
                progress.print(f"[yellow]Skipping frame #{index}/{reference_video.TotalFramesCount-1} as it already exists at {save_filepath.absolute()}")
                progress.advance(overallProcessTask, 1)
            index += 1
            continue

        frame = utils.resize_image(Image.fromarray(frame), resize_factor)
        tiled_frame = generate_tiledimage_gu(np.asarray(frame), tiles, tile_shape, useCuda)
        Image.fromarray(tiled_frame).save(save_filepath)
        index += 1

        if progress:
            progress.advance(overallProcessTask, 1)


    if progress:
        progress.update(overallProcessTask, description=f"Finished processing {reference_video.path.name}",
                        total=index)
        progress.print(f"Finished processing in {timer.getTimeSinceStart()}s")

def run_ffmpeg(progress: Progress,pattern: Path,save_path: Path="./out.mp4", fps: int=30):
    try:

        results = subprocess.run(f"ffmpeg -framerate {fps} -hwaccel auto -i \"{pattern.absolute()}\" \"{save_path.absolute()}\"",
                                 capture_output=True,
                                 input="y".encode())
    except Exception as e:
        progress.print(f"[red]Failed to run ffmpeg:\n{e}")
        progress.stop()
    finally:
        progress.print(f"ffmpeg outputs:\n{results.stderr.decode()}")
        progress.print(f"ffmpeg return code: {results.returncode}")

        if results.returncode != 0:
            progress.print("[red]ffmpeg failed!")
            progress.print(f"[yellow]Arguments: \n{results.args}")


def video_cli(
        source_path: Path = typer.Argument(..., help="Path to source video to use as reference"),
        save_path: Path = typer.Argument(..., help="Path to save the final result to. Eg ./out.png"),
        tileset_paths: list[Path] = typer.Argument(...,
                                                   help="Path to images used as tiles. Eg: './assets/tiles/*.png' or './assets/tiles/a.png ./assets/tiles/n.png' ..."),
        resize_factor: float = typer.Option(-1,
                                            help="Resize factor for reference image, so that the final image is not too big. Default: -1 (resizes based on tile size)"),
        process_type: ProcessType = typer.Option(ProcessType.guvectorize,
                                                 help="Type of processing to use. Default: guvectorize. njit IS not available for video")
):
    overall_progress = TilImg.utils.getProgressBar()
    overall_progress_task = overall_progress.add_task(f"Overall Progress", total=4)



    if process_type == ProcessType.njit:
        rich.print(
            f"[red]Invalid process type[/red]: {ProcessType.njit} is not available for video processing. Please use {ProcessType.guvectorize} or {ProcessType.cuda} instead.")
        return

    with overall_progress:
        useCuda = process_type == ProcessType.cuda

        tiles, tile_shape = TilImg.utils.load_imageset(Path(), "", tileset_paths, progress=overall_progress)
        overall_progress.advance(overall_progress_task, 1)
        video = TilImg.video.Video(source_path)
        overall_progress.print(f"[yellow]Final video resolution: {tiles.shape[1]*video.width}x{tiles.shape[0]*video.height}")
        TilImg.video.generate_tiledimage_video(video, tiles, tile_shape, useCuda=useCuda,
                                               resize_factor=resize_factor, progress=overall_progress)
        overall_progress.advance(overall_progress_task, 1)
        utils.test_for_ffmpeg()
        overall_progress.update(overall_progress_task, description="Overall Progress - waiting for ffmpeg...",advance=1)
        run_ffmpeg(overall_progress, Path(f"./build/frames/{source_path.name}_%d.png"), save_path, video.Framerate)
        overall_progress.update(overall_progress_task, description="Overall Progress - Finished",advance=1)
        overall_progress.print(f"Saved to f{save_path.absolute()}")