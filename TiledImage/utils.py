import enum
import math
import subprocess
import time
from pathlib import Path
from typing import Union

import numpy as np
import tqdm
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeRemainingColumn, BarColumn, MofNCompleteColumn, \
    TimeElapsedColumn

from TiledImage import UnexpectedImageShapeError


class ClockTimer:
    def __init__(self):
        self.start_ = 0
        self.last = 0

    def start(self):
        self.start_ = time.perf_counter()
        self.last = self.start_

    def getTimeSinceLast(self):
        now = time.perf_counter()
        delta = now - self.last
        self.last = now
        return delta

    def getTimeSinceStart(self):
        return time.perf_counter() - self.start_


class SpinnerProgress(Progress):
    def __init__(self, *args, **kwargs):
        Progress.__init__(self, SpinnerColumn(finished_text=""), TextColumn("[progress.description]{task.description}"),
                          *args, **kwargs)

def getProgressBar():
    return Progress(SpinnerColumn(),"{task.description}", BarColumn(), MofNCompleteColumn(),
                                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(),
                                TimeRemainingColumn())

def load_image(path: Path, resize: Union[float, tuple[int, int]] = 1, keep_ratio: bool = True,
               progress:Union[Progress,None]=None) -> np.ndarray:
    """
    Load an image from a path
    :param path:
    :param resize: Resizes image, if float, resizes by that factor. When tuple, is max size (will resize to below or equal)
    :param keep_ratio: When resizing, whether to keep the aspect ratio. Does not apply when resizing by a float
    :param silent: Whether to print debug statements
    :return:
    """
    im = Image.open(path).convert("RGBA")
    originalSize = im.size
    im = resize_image(im, resize, keep_ratio)
    if progress:
        progress.print(f"[green]Loaded reference image {path} with original size {originalSize} and resized to {im.size}")

    return np.asarray(im)


def resize_image(im: Image.Image, resize: Union[float, tuple[int, int]] = 1, keep_ratio: bool = True, ) -> Image.Image:
    """
    Resizes an image to a given size
    :param im: Image to resize
    :param resize: Resizes image, if float, resizes by that factor. When tuple, is max size (will resize to below or equal)
    :param keep_ratio: When resizing, whether to keep the aspect ratio. Does not apply when resizing by a float
    :return:
    """
    if resize != 1:
        if isinstance(resize, float):
            im = im.resize((int(im.width * resize), int(im.height * resize)), resample=Image.NEAREST)

        elif isinstance(resize, tuple):
            if keep_ratio:
                ratio = min(resize[0] / im.width, resize[1] / im.height)
                im = im.resize((int(im.width * ratio), int(im.height * ratio)), resample=Image.NEAREST)
            else:
                im = im.resize(resize, resample=Image.NEAREST)
        else:
            raise TypeError(f"resize must be a float or tuple, not {type(resize)}")

    return im


def load_imageset(imageSetDirectory: Path, glob: str = "*.png", image_paths: Union[None, list[Path]] = None,
                  progress: Union[Progress,None] = None) -> tuple[np.ndarray, tuple[int]]:
    """
    Load all images in a directory and return a list of images

    :param imageSetDirectory: The directory to load images from
    :param glob: The glob to use when loading images
    :param image_paths: The path to al the images. when not none, will ignore imageSetDirectory and glob and use this instead
    :return:
    """

    if image_paths is None:
        if not imageSetDirectory.exists():
            raise FileNotFoundError(f"Path {imageSetDirectory} does not exist")
        if not imageSetDirectory.is_dir():
            raise FileNotFoundError(f"{imageSetDirectory} is not a directory")

        if progress:
            progress.console.print(f"Loading image set from directory '{imageSetDirectory}/{glob}'")

        image_paths = list(imageSetDirectory.glob(glob))

    if progress:
        task = progress.add_task("Loading image set for tiles", total=len(image_paths))

    image_set = []
    image_shape = None

    for file in image_paths:

        im = load_image(file)

        if image_shape is None:
            image_shape = im.shape
            if progress:
                progress.console.print(f"Using {file} image's shape as expected shape: {image_shape}")

        if image_shape != im.shape:
            raise UnexpectedImageShapeError(f"Image {file} has shape {im.shape} but expected {image_shape}")

        image_set.append(im)
        if progress:
            progress.update(task, description=f"Loading image set for tiles - {file.name}", advance=1)
    if progress:
        progress.update(task, description=f"Loaded image set for tiles [Done]", advance=1)
    return np.array(image_set), image_shape


def create_tiles_atlas(imageSet: list[np.ndarray], shape: tuple[int]) -> np.ndarray:
    """
    Composites the indivdual images together to form a giant image set atlas

    :param imageSet:
    :param shape: The shape of a single image in the image set. If set incorrectly, will result in overlapping. Only first two dimensions are used
    :return:
    """
    print("Creating tiles atlas...")
    imageSetCount = len(imageSet)
    side_count = int(math.ceil(math.sqrt(imageSetCount)))
    atlas_shape = (side_count * shape[0], side_count * shape[1], 4)
    atlas = np.zeros(atlas_shape, dtype=np.uint8)

    t_ = tqdm.tqdm(np.ndindex((side_count, side_count)), total=imageSetCount, desc="Creating tiles atlas...")

    for ix, iy in t_:
        index = ix * side_count + iy
        x, y = ix * shape[0], iy * shape[1]
        try:
            atlas[x:x + shape[0], y:y + shape[1], :] = imageSet[index]
        except IndexError:
            break

    return atlas


def manual_shape_correction(arr: np.ndarray):
    """
    Manually corrects the shape of the numpy array returned by tile_pixel_compare.
    This is because when using numpy reshape, it will cause each tile to be shifted up by one pixel incrementally
    (So the first tile will be shifted up by 1 pixel, the second by 2, etc)

    :param arr: Array to correct. Has to have 5 dimensions
    :return:
    """

    if len(arr.shape) != 6:
        raise ValueError(f"Array must have 6 dimensions, not {len(arr.shape)}")

    width = arr.shape[0] * arr.shape[3]
    height = arr.shape[1] * arr.shape[4]
    image = np.zeros((width, height, 4), dtype=np.uint8)
    print(image.shape, arr.shape, flush=True)
    _tqdm = tqdm.tqdm(np.ndindex(arr.shape[:2]), total=arr.shape[0] * arr.shape[1], desc="Correcting shape")
    for ix, iy in _tqdm:
        x = ix * arr.shape[3]
        y = iy * arr.shape[4]
        image[x:x + arr.shape[3], y:y + arr.shape[4], :] = arr[ix, iy, 3]

    return image


class ProcessType(str, enum.Enum):
    njit = "njit",
    guvectorize = "guvectorize",
    cuda = "cuda"

def test_for_ffmpeg():
    """
    Tests for ffmpeg in the path
    :return:
    """
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True)
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg not found in path. Please install ffmpeg and add it to your path!")

