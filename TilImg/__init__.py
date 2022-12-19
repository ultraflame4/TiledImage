__version__ = "3.1.5"

import numpy as np
import numba as nb
from rich.progress import Progress, SpinnerColumn, TextColumn

from TilImg.errors import UnexpectedImageShapeError
from TilImg.utils import ClockTimer, SpinnerProgress

nb.warnings.simplefilter('ignore', category=nb.NumbaDeprecationWarning)


def _tile_pixel_compare(refPixel: np.ndarray, tiles_avg: np.ndarray, out: int):
    nearest_tile_index:nb.int64 = 0
    nearest_distance = -1
    c:nb.int64 = 0
    # print("a",refPixel.shape)
    for t in tiles_avg:

        distance = (refPixel[0] - t[0])**2+(refPixel[1] - t[1])**2+(refPixel[2] - t[2])**2+(refPixel[3] - t[3])**2

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
    Computes the average pixel of each tile in the tiles array

    :param tiles: A 4D numpy array representing the tiles. Aka, and array of images
    :return:
    """

    return (np.mean(tiles, axis=(1, 2))+np.median(tiles, axis=(1, 2)))/2


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


def generate_tiledimage_gu(reference: np.ndarray, tiles: np.ndarray, tile_shape: tuple[int],useCuda=False, progress:Progress=None) -> np.ndarray:
    """
    Generates a single channel image from a reference image and a set of tiles

    :param reference: A 3D numpy array representing the reference image
    :param tiles: An 4D numpy array representing the tiles. Aka, and array of images
    :param tile_shape: Shape of a single tile
    :param channel: Which pixel channel to use, 0=Red, 1=Green, 2=Blue, 3=Alpha
    :return:
    """

    if progress:
        progress.console.print("Generating using numba's @guvectorize() !")

        if useCuda:
            progress.console.print("Using Nvidia's CUDA!!!")


    tile_shape=np.asarray(tile_shape)

    clock = ClockTimer()
    clock.start()

    if progress:
        avg_tiles_task = progress.add_task("[1/3] Computing tiles average...", total=1)
        compare_task = progress.add_task("[2/3]. Comparing tiles and pixels...", total=1)
        draw_task = progress.add_task("[3/3]. Drawing final results ...", total=1)

    avg_tiles = compute_tiles_avg(tiles)
    if progress:
        progress.update(avg_tiles_task, advance=1, description=f"[1/3] Completed Computing tiles average in {clock.getTimeSinceLast()}s")



    if useCuda:
        result = tile_pixel_compare_cuda(reference, avg_tiles)
    else:
        result = tile_pixel_compare(reference, avg_tiles)

    if progress:
        progress.update(compare_task, advance=1, description=f"[2/3] Completed comparing tiles and pixels in {clock.getTimeSinceLast()}s")

    final = draw_final_results_gu(result, tiles, tile_shape)
    if progress:
        progress.update(draw_task, advance=1, description=f"[3/3] Draw completed in {clock.getTimeSinceLast()}s")

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


    print("[1/3]. Computing tiles average...")
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
