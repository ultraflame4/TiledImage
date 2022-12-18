import os
from pathlib import Path

import colorama
import numpy
import numpy as np
import cv2
import tqdm
from PIL import Image

from TiledImage import ClockTimer, generate_tiledimage_gu, utils


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


def generate_tiledimage_video(reference_video: Video, tiles: np.ndarray, tile_shape: tuple[int]):

    print(f"Generating tiled image from video with {reference_video.path}")
    print(f"Video frames count: {reference_video.TotalFramesCount}")
    print(f"Video framerate count: {reference_video.Framerate}")
    print(f"Tiles shape: {tiles.shape}")
    os.makedirs(Path("./build/frames"),exist_ok=True)


    index = 0
    for frame in reference_video.getFrame():
        print(f"{colorama.Fore.CYAN}>>>> Processing {reference_video.path} Frame [{index}/{reference_video.TotalFramesCount}] >>>>{colorama.Fore.RESET}")
        save_filepath=Path(f"./build/frames/{index}.png")
        if save_filepath.exists():
            print(f"{colorama.Fore.YELLOW}>>>> Frame [{index}/{reference_video.TotalFramesCount}] already exits at {save_filepath}! Skipping... >>>>{colorama.Fore.RESET}")
            continue
        print("Resizing frame...")
        frame = utils.resize_image(Image.fromarray(frame), 1 / max(tile_shape))
        tiled_frame = generate_tiledimage_gu(np.asarray(frame), tiles, tile_shape)
        Image.fromarray(tiled_frame).save(save_filepath)
        index += 1
        print(f"{colorama.Fore.GREEN}>>>>Finished {reference_video.path} Frame  [{index}/{reference_video.TotalFramesCount}] >>>>{colorama.Fore.RESET}")

    print(f"{colorama.Fore.LIGHTGREEN_EX}>>>> Finished processing {reference_video.path} >>>>>{colorama.Fore.RESET}")

