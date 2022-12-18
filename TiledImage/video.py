import os
from pathlib import Path
from typing import Union

from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table

import TiledImage
import colorama
import numpy
import numpy as np
import cv2
import rich
import tqdm
import typer
from PIL import Image

from TiledImage import ClockTimer, generate_tiledimage_gu, utils
from TiledImage.utils import ProcessType


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

    os.makedirs(Path("./build/frames"), exist_ok=True)

    timer = ClockTimer()
    timer.start()
    index = 0

    for frame in reference_video.getFrame():

        save_filepath = Path(f"./build/frames/{reference_video.path.name}_{index}.png")
        if save_filepath.exists():
            if progress:
                progress.print(f"[yellow]Skipping {save_filepath} as it already exists")
                progress.advance(overallProcessTask, 1)
            index += 1
            continue

        frame = utils.resize_image(Image.fromarray(frame), 1 / max(tile_shape))
        tiled_frame = generate_tiledimage_gu(np.asarray(frame), tiles, tile_shape, useCuda)
        # Image.fromarray(tiled_frame).save(save_filepath)
        index += 1

        progress.advance(overallProcessTask, 1)

    if progress:
        progress.update(overallProcessTask, description=f"Finished processing {reference_video.path.name}",
                        total=index)


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
    overall_progress = Progress("{task.description}", BarColumn(), MofNCompleteColumn(),
                                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(),
                                TimeRemainingColumn())
    overall_progress_task = overall_progress.add_task(f"Overall Progress", total=3)

    progress_table = Table.grid()
    progress_table.add_row(overall_progress)

    if process_type == ProcessType.njit:
        rich.print(
            f"[red]Invalid process type[/red]: {ProcessType.njit} is not available for video processing. Please use {ProcessType.guvectorize} or {ProcessType.cuda} instead.")
        return

    with Live(progress_table, refresh_per_second=10, ):
        useCuda = process_type == ProcessType.cuda

        tiles, tile_shape = TiledImage.utils.load_imageset(Path(), "", tileset_paths, progress=overall_progress)
        overall_progress.advance(overall_progress_task, 1)
        video = TiledImage.video.Video(source_path)
        TiledImage.video.generate_tiledimage_video(video, tiles, tile_shape, useCuda=useCuda,
                                                   resize_factor=resize_factor, progress=overall_progress)
        overall_progress.advance(overall_progress_task, 1)
