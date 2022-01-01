__version__ = "2.0.0.dev1"

# Flags
import dataclasses
import glob
import math
import os.path
import time

import cv2
import numpy as np
import numba as nb
from numba import uint8
import tqdm

from TiledImage import core


@nb.guvectorize([(uint8[:, :, :, :], uint8[:, :])], "(n,a,b,c)->(n,c)", target='cpu')
def compute_avg_tile_vals(tiles: np.ndarray, outarray: np.ndarray):
    for index in range(tiles.shape[0]):
        tile = tiles[index]
        outarray[index] = np.array([
            np.mean(tile[:, :, 0]),
            np.mean(tile[:, :, 1]),
            np.mean(tile[:, :, 2])
            ])


def normal_compute(refImage: np.ndarray, outImage: np.ndarray, tileAvgVals: np.ndarray, tiles: np.ndarray,
                   tileShape: tuple):

    def getDist(a:np.ndarray,b:np.ndarray):
        return math.sqrt((float(a[0]) - float(b[0])) ** 2 +
                         (float(a[1])-float(b[1])) ** 2 + (float(a[2]) - float(b[2])) ** 2)

    for y, x in nb.pndindex(refImage.shape[:2]):
        refPixel = refImage[y, x]
        nx, ny = x * tileShape[1], y * tileShape[0]
        # get closest color
        tileIndex = 0
        distance = 1000
        for i in nb.prange(tileAvgVals.shape[0]):
            c = tileAvgVals[i]
            dist = getDist(c, refPixel)
            if dist < distance:
                distance = dist
                tileIndex = i

        t = tiles[tileIndex]
        for ty in range(t.shape[0]):
            for tx in range(t.shape[1]):
                for channel in range(3):
                    outImage[ny + ty:ny + ty + 1, nx + tx:nx + tx + 1, channel] = t[ty, tx, channel]


def generate(reference_path: str, tilesdir: str, output_path: str, timeit: bool = False, max_memory: int = 2048,
             scale_down=True, compute: core.c_option = "normal", verbose: bool = False):
    ti = core.TiledImageGenerate()
    ti.flags = core.OptionFlags(reference_path, tilesdir, output_path, timeit, max_memory, compute)

    if ti.flags.COMPUTE_OPTION == "cuda":
        print("Warning: Only CUDA GPUs are supported. The CUDA toolkit must also be installed! "
              "Get CUDA toolkit from: https://developer.nvidia.com/cuda-toolkit")

    print(f"Loading tiles from {tilesdir} ...")
    tilesdir = tilesdir + "/*"
    ti.tiles = np.array([cv2.imread(f) for f in tqdm.tqdm(glob.iglob(tilesdir))])  # load tiles
    if ti.tiles.shape[0] < 1:
        raise ValueError(f"There is less than one tile in tile directory {tilesdir}")

    print(f"Loading Reference Image from {reference_path} ...")
    ti.reference = cv2.imread(reference_path)
    if ti.reference is None:
        raise ValueError(f"Invalid reference image path {reference_path}")
    sample_tile: np.ndarray = ti.tiles[0]
    ti.tile_shape = sample_tile[0], sample_tile[1]
    if scale_down:
        if verbose: print("Scaling down reference image")
        ti.reference = cv2.resize(ti.reference,
                                  (
                                      math.floor(ti.reference.shape[1] / sample_tile.shape[1]),
                                      math.floor(ti.reference.shape[0] / sample_tile.shape[0])
                                      )
                                  )
        if verbose: print(f"Resulting reference image size {ti.reference.shape}")

    if verbose: print("Creating canvas...")
    ti.canvas = np.zeros((
        ti.reference.shape[0] * sample_tile.shape[0],
        ti.reference.shape[1] * sample_tile.shape[1],
        3
        ), dtype=ti.reference.dtype)
    if verbose: print("Canvas size", ti.canvas.shape)

    print("Computing tile average color values...")
    ti.average_tile_vals = np.zeros((ti.tiles.shape[0], 3), dtype=ti.reference.dtype)
    compute_avg_tile_vals(ti.tiles, ti.average_tile_vals)

    # no need to multiply by item size. item size is 1byte as dtype is uint8
    bytes_left = ti.flags.MAX_MEMORY * 1000 - (ti.tiles.size + ti.average_tile_vals.size)
    if (bytes_left < 1024):
        raise MemoryError(
            f"Out of memory to use! There is less than 1MB of memory to work with! Please increase max memory! Current: max memory {max_memory} MB")

    wc_size = math.floor(math.sqrt(bytes_left))
    ti.set_wc(wc_size)
    if verbose: print(f"Using {wc_size ** 2}bytes of memory for working canvases ({wc_size}x{wc_size})")

    itershape = math.ceil(ti.reference.shape[1] / wc_size), math.ceil(ti.reference.shape[0] / wc_size)
    with tqdm.tqdm(total=itershape[0] * itershape[1], desc="Computing final image...") as bar:
        for x, y in np.ndindex(*itershape):
            # a portion of the whole reference image to send to compute functions
            ref_slice = ti.reference[
                        y * wc_size:y * wc_size + wc_size,
                        x * wc_size:x * wc_size + wc_size,
                        ]
            ti.working_canvas = np.zeros((
                ref_slice.shape[0] * sample_tile.shape[0],
                ref_slice.shape[1] * sample_tile.shape[1],
                3
                ), dtype=np.uint8)
            normal_compute(ref_slice, ti.working_canvas, ti.average_tile_vals, ti.tiles, sample_tile.shape)
            # paste onto canvas

            ti.canvas[
            y * ti.working_canvas.shape[0]:y * ti.working_canvas.shape[0] + ti.working_canvas.shape[0],
            x * ti.working_canvas.shape[1]:x * ti.working_canvas.shape[1] + ti.working_canvas.shape[1]] = ti.working_canvas
            bar.update()

    cv2.imwrite(ti.flags.OUT_PATH,ti.canvas)
    print("Saved results to", ti.flags.OUT_PATH)
