import os
from pathlib import Path

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

    def getFrame(self)->np.ndarray:
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            yield frame

    def __del__(self):
        self.cap.release()


def generate_tiledimage_video(reference_video: Video, tiles: np.ndarray, tile_shape: tuple[int],useCuda=False):

    print(f"Generating tiled image from video with {reference_video.path}")
    print(f"Video frames count: {reference_video.TotalFramesCount}")
    print(f"Video framerate count: {reference_video.Framerate}")
    print(f"Tiles shape: {tiles.shape}")
    os.makedirs(Path("./build/frames"),exist_ok=True)

    timer = ClockTimer()
    timer.start()
    index = 0
    print("\n"*64,end="")
    for frame in reference_video.getFrame():
        print(f"{colorama.Cursor.POS(0,0)}{colorama.ansi.clear_line()}",end="")
        print(f"{colorama.Fore.CYAN}>>>> Processing {reference_video.path} Frame [{index}/{reference_video.TotalFramesCount}] >>>>{colorama.Fore.RESET}")
        save_filepath=Path(f"./build/frames/{index}.png")
        if save_filepath.exists():
            print(f"{colorama.Fore.YELLOW}>>>> Frame [{index}/{reference_video.TotalFramesCount}] already exits at {save_filepath}! Skipping... >>>>{colorama.Fore.RESET}")
            index+=1
            continue
        print("Resizing frame...")
        frame = utils.resize_image(Image.fromarray(frame), 1 / max(tile_shape))
        tiled_frame = generate_tiledimage_gu(np.asarray(frame), tiles, tile_shape,useCuda)
        Image.fromarray(tiled_frame).save(save_filepath)
        index += 1
        print(f"{colorama.Fore.GREEN}>>>>Finished and Saved Frame {index}/{reference_video.TotalFramesCount} in {timer.getTimeSinceLast()}s>>>>{colorama.Fore.RESET}")

    print(f"{colorama.Fore.LIGHTGREEN_EX}>>>> Finished processing {reference_video.path} >>>>>{colorama.Fore.RESET}")


def video_cli(
        source_path: Path = typer.Argument(..., help="Path to source video to use as reference"),
        save_path: Path = typer.Argument(..., help="Path to save the final result to. Eg ./out.png"),
        tileset_paths: list[Path] = typer.Argument(..., help="Path to images used as tiles. Eg: './assets/tiles/*.png' or './assets/tiles/a.png ./assets/tiles/n.png' ..."),
        resize_factor: float = typer.Option(-1, help="Resize factor for reference image, so that the final image is not too big. Default: -1 (resizes based on tile size)"),
        process_type: ProcessType = typer.Option(ProcessType.guvectorize, help="Type of processing to use. Default: guvectorize. njit IS not available for video")
              ):

    if process_type == ProcessType.njit:
        rich.print(f"[red]Invalid process type[/red]: {ProcessType.njit} is not available for video processing. Please use {ProcessType.guvectorize} or {ProcessType.cuda} instead.")
        return


    pass