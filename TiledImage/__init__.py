__version__ = "3.0.0"

import math
import time
from pathlib import Path
from typing import Union

import numpy as np
import tqdm
from PIL import Image
import numba as nb
from TiledImage.errors import UnexpectedImageShapeError
from TiledImage.utils import ClockTimer


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
    image = np.zeros((width, height,4), dtype=np.uint8)
    print(image.shape,arr.shape,flush=True)
    _tqdm = tqdm.tqdm(np.ndindex(arr.shape[:2]), total=arr.shape[0] * arr.shape[1], desc="Correcting shape")
    for ix, iy in _tqdm:
        x = ix * arr.shape[3]
        y = iy * arr.shape[4]
        image[x:x + arr.shape[3], y:y + arr.shape[4],:] = arr[ix, iy,3]

    return image


def _tile_pixel_compare(refPixel: np.ndarray, tiles_avg: np.ndarray, out: int):
    nearest_tile_index:nb.int64 = 0
    nearest_distance = -1
    c:nb.int64 = 0
    # print("a",refPixel.shape)
    for t in tiles_avg:

        distance = (refPixel[0] - t[0])**2+(refPixel[1] - t[1])**2+(refPixel[2] - t[2])**2

        if nearest_distance == -1 or distance < nearest_distance:
            nearest_distance = distance
            nearest_tile_index = c
        c += 1

    out[:] = nearest_tile_index

tile_pixel_compare= nb.guvectorize([(nb.uint8[:],
                  nb.float64[:,:],
                  nb.int64[:])],
                "(x),(a,b)->(x)", nopython=True, target="parallel")(_tile_pixel_compare)

tile_pixel_compare_cuda= nb.guvectorize([(nb.uint8[:],
                  nb.float64[:,:],
                  nb.int64[:])],
                "(x),(a,b)->(x)", nopython=True, target="cuda")(_tile_pixel_compare)


def compute_tiles_avg(tiles: np.ndarray) -> np.ndarray:
    """
    Computes the mean of each tile in the tiles array

    :param tiles: A 4D numpy array representing the tiles. Aka, and array of images
    :return:
    """

    return np.mean(tiles, axis=(1, 2))


@nb.njit(fastmath=True, parallel=True)
def compute_tileAvg_vs_referencePixel(pixel: np.ndarray, tilesPixelsAvg: np.ndarray) -> int:
    """

    :param pixel: A single 1d array with length 3 representing a pixel
    :param tilesPixelsAvg: A 2d array with shape (n, 3) representing a list of the tiles' avg pixel values
    :return:
    """
    closest_tile_index = 0
    closest_tile_dist = -1
    index = 0
    for tile in tilesPixelsAvg:
        _dist = np.linalg.norm(pixel - tile)
        if closest_tile_dist == -1 or _dist < closest_tile_dist:
            closest_tile_dist = _dist
            closest_tile_index = index
        index += 1
    return closest_tile_index


@nb.njit(fastmath=True, parallel=True)
def computeClosestTilePixelMatches(reference: np.ndarray, tilesAvg: np.ndarray) -> np.ndarray:
    """

    :param reference: A 3D numpy array representing the reference image
    :param tilesPixelsAvg: A 2d array with shape (n, 3) representing a list of the tiles' avg pixel values
    :return:
    """
    closest_matches = np.zeros(reference.shape[:2], dtype=np.uint)

    h = reference.shape[0]
    w = reference.shape[1]
    l = h * w

    for i in nb.prange(l):
        x = i % w
        y = i // w
        closest_matches[y, x] = compute_tileAvg_vs_referencePixel(reference[y, x], tilesAvg)

    # Slower
    # for ix, iy in np.ndindex(reference.shape[:2]):
    #     closest_matches[ix, iy] = compute_tileAvg_vs_referencePixel(reference[ix, iy], tilesAvg)

    return closest_matches


@nb.njit(fastmath=True, parallel=True)
def draw_final_results(closest_matches: np.ndindex, tiles: np.ndarray, tile_shape: np.ndarray) -> np.ndarray:
    canvas = np.zeros((closest_matches.shape[0] * tile_shape[0], closest_matches.shape[1] * tile_shape[1], 4),
                      dtype=np.uint8)
    h = closest_matches.shape[0]
    w = closest_matches.shape[1]
    l = h * w

    for i in nb.prange(l):
        x = i % w
        y = i // w

        canvas[
        y * tile_shape[0]:(y + 1) * tile_shape[0],
        x * tile_shape[1]:(x + 1) * tile_shape[1],
        :
        ] = tiles[closest_matches[y, x]]

    return canvas
@nb.njit(fastmath=True, parallel=True)
def draw_final_results_gu(closest_matches: np.ndindex, tiles: np.ndarray, tile_shape: np.ndarray) -> np.ndarray:
    canvas = np.zeros((closest_matches.shape[0] * tile_shape[0], closest_matches.shape[1] * tile_shape[1], 4),
                      dtype=np.uint8)
    h = closest_matches.shape[0]
    w = closest_matches.shape[1]
    l = h * w

    for i in nb.prange(l):
        x = i % w
        y = i // w

        canvas[
        y * tile_shape[0]:(y + 1) * tile_shape[0],
        x * tile_shape[1]:(x + 1) * tile_shape[1],
        :
        ] = tiles[closest_matches[y, x][0]]

    return canvas


def generate_tiledimage_gu(reference: np.ndarray, tiles: np.ndarray, tile_shape: tuple[int],useCuda=False) -> np.ndarray:
    """
    Generates a single channel image from a reference image and a set of tiles

    :param reference: A 3D numpy array representing the reference image
    :param tiles: An 4D numpy array representing the tiles. Aka, and array of images
    :param tile_shape: Shape of a single tile
    :param channel: Which pixel channel to use, 0=Red, 1=Green, 2=Blue, 3=Alpha
    :return:
    """
    print("Generating using numba's @guvectorize() !")
    if useCuda:
        print("Using Nvidia's CUDA!!!")
    tile_shape=np.asarray(tile_shape)

    clock = ClockTimer()
    clock.start()

    print("[1/3]. Computing tiles average (mean)...")
    avg_tiles = compute_tiles_avg(tiles)
    print("[1/3]. Completed in", clock.getTimeSinceLast(), "s")

    print("[2/3]. Comparing tiles and pixels...")
    if useCuda:
        result = tile_pixel_compare_cuda(reference, avg_tiles)
    else:
        result = tile_pixel_compare(reference, avg_tiles)
    print("[2/3]. Completed in", clock.getTimeSinceLast(), "s")

    print("[3/3]. Drawing final results ...")
    final = draw_final_results_gu(result, tiles, tile_shape)
    print("[3/3]. Completed in", clock.getTimeSinceLast(), "s")
    print("[///]. Total time taken", clock.getTimeSinceStart(), "s")

    return final

def generate_tiledimage(reference: np.ndarray, tiles: np.ndarray, tile_shape: tuple[int]) -> np.ndarray:
    """
    The main function for generating a tiled image from a reference image and a set of tiles.

    :param reference:
    :param tiles:
    :param tile_shape:
    :return:
    """

    clock = ClockTimer()
    clock.start()


    print("[1/3]. Computing tiles average (mean)...")
    avg_vals = compute_tiles_avg(tiles)
    print("[1/3]. Completed in", clock.getTimeSinceLast(), "s")

    print("[2/3]. Computing closest tile-pixel matches ...")
    pixelTileMatches = computeClosestTilePixelMatches(reference, avg_vals)
    print("[2/3]. Completed in", clock.getTimeSinceLast(), "s")

    print("[3/3]. Drawing final results ...")
    final = draw_final_results(pixelTileMatches, tiles, tile_shape)
    print("[3/3]. Completed in", clock.getTimeSinceLast(), "s")
    print("[///]. Total time taken", clock.getTimeSinceStart(), "s")
    return final
    # channels_count = 3
    # i_ = tqdm.tqdm(range(4), desc="Generating channels")
    # results = []
    # for i in i_:
    #     i_.set_description(f"Generating channel {i}")
    #     results.append(generate_singlechannel(reference, tiles, tile_shape, i))
    #
    # return np.dstack(results)
