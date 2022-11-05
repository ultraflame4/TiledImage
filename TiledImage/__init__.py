import math
from pathlib import Path
from typing import Union

import numpy as np
import tqdm
from PIL import Image
import numba as nb
from TiledImage.errors import UnexpectedImageShapeError


def load_image(path: Path, resize: Union[float, tuple[int, int]] = 1, keep_ratio: bool = True,
               silent=True) -> np.ndarray:
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

    if not silent:
        print(f"Loaded image {path} with original size {originalSize} and resized to {im.size}")

    return np.asarray(im)


def load_imageset(imageSetDirectory: Path, glob: str = "*.png") -> tuple[np.ndarray, tuple[int]]:
    """
    Load all images in a directory and return a list of images
    :param imageSetDirectory:
    :return:
    """
    if not imageSetDirectory.exists():
        raise FileNotFoundError(f"Path {imageSetDirectory} does not exist")
    if not imageSetDirectory.is_dir():
        raise FileNotFoundError(f"{imageSetDirectory} is not a directory")

    print(f"\nLoading image set from directory '{imageSetDirectory}/{glob}'")
    image_paths = list(imageSetDirectory.glob(glob))
    tqdm_ = tqdm.tqdm(image_paths, desc="Loading image set")
    image_set = []
    image_shape = None

    for file in tqdm_:
        tqdm_.set_description(f"Loading {file.name}")
        im = load_image(file)

        if image_shape is None:
            image_shape = im.shape
            tqdm_.write(f"Using {file} image's shape as expected shape: {image_shape}")

        if image_shape != im.shape:
            raise UnexpectedImageShapeError(f"Image {file} has shape {im.shape} but expected {image_shape}")

        image_set.append(im)

    print("Successfully loaded image set\n")

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




@nb.guvectorize([(nb.uint8, nb.uint8[:, :, :], nb.int64[:], nb.uint8[:, :], nb.uint8[:, :])],
                "(),(a,b,c),(s),(e,f)->(e,f)", nopython=True)
def tile_pixel_compare(refPixel: np.ndarray, tiles: np.ndarray, tile_shape: np.ndarray, canvas_: np.ndarray,
                       out: np.ndarray) -> np.ndarray:

    nearest_tile_index = 0
    nearest_distance = -1
    c =0
    for t in tiles:
        tile_mean= np.mean(t)
        distance=abs(tile_mean-refPixel)

        if nearest_distance == -1 or distance < nearest_distance:
            nearest_distance = distance
            nearest_tile_index = c
        c+=1
    # print(tiles[nearest_tile_index])
    out[:,:] = tiles[nearest_tile_index]
    pass


def manual_shape_correction(arr: np.ndarray):
    """
    Manually corrects the shape of the numpy array returned by tile_pixel_compare.
    This is because when using numpy reshape, it will cause each tile to be shifted up by one pixel incrementally
    (So the first tile will be shifted up by 1 pixel, the second by 2, etc)

    :param arr: Array to correct. Has to have 5 dimensions
    :return:
    """
    if len(arr.shape) != 4:
        raise ValueError(f"Array must have 4 dimensions, not {len(arr.shape)}")

    width = arr.shape[0] * arr.shape[2]
    height = arr.shape[1] * arr.shape[3]
    image = np.zeros((width, height), dtype=np.uint8)
    _tqdm = tqdm.tqdm(np.ndindex(arr.shape[:2]),total=arr.shape[0]*arr.shape[1], desc="Correcting shape")

    for ix, iy in _tqdm:
        x = ix * arr.shape[2]
        y = iy * arr.shape[3]
        image[x:x + arr.shape[2], y:y + arr.shape[3]] = arr[ix, iy]

    return image



def generate_singlechannel(reference_: np.ndarray, tiles_: np.ndarray, tile_shape: tuple[int],channel=0) -> np.ndarray:
    """
    Generates a single channel image from a reference image and a set of tiles

    :param reference_: A 3D numpy array representing the reference image
    :param tiles_: An 4D numpy array representing the tiles. Aka, and array of images
    :param tile_shape: Shape of a single tile
    :param channel: Which pixel channel to use, 0=Red, 1=Green, 2=Blue, 3=Alpha
    :return:
    """
    print(f"Generating single channel ({channel}) image from reference image...")
    reference=reference_[:,:,channel]
    tiles=tiles_[:,:,:,channel]

    canvasShape = (reference.shape[0] * tile_shape[0], reference.shape[1] * tile_shape[1])
    print(f"\tWill expect result of shape: {canvasShape} with tile shape {tile_shape} and reference shape {reference.shape}")

    result = tile_pixel_compare(reference, tiles, np.asarray(tile_shape), np.zeros(tile_shape[:2], dtype=np.uint8))
    print(f"\tReshaping result of shape {result.shape} to {canvasShape}")
    # return canvas
    return manual_shape_correction(result)


def generate_tiledimage(reference: np.ndarray, tiles: np.ndarray, tile_shape: tuple[int]) -> np.ndarray:
    """
    The main function for generating a tiled image from a reference image and a set of tiles.

    :param reference:
    :param tiles:
    :param tile_shape:
    :return:
    """

    channels_count = 4
    i_=tqdm.tqdm(range(4),desc="Generating channels")
    for i in i_:
        i_.set_description(f"Generating channel {i}")
        result = generate_singlechannel(reference, tiles, tile_shape, i)



    pass