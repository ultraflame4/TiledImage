"""
This module contains stuff that tiles and generates the tiled image faster
"""
import glob
import math
from typing import Literal

import cv2
import numba
import numpy as np
import tqdm
import numba as nb
from numba import cuda

from TiledImage.old import others


def loadTiles(path: str) -> np.ndarray:
    others.printLoadingTiles()
    return np.array([cv2.imread(f) for f in tqdm.tqdm(glob.iglob(path))])


@numba.jit(nopython=True, parallel=True)
def computeTileAverageValues(tiles: np.ndarray):
    print("Computing Average color for each tile")
    avgToTile: np.ndarray = np.zeros((tiles.shape[0], 3))
    for i in nb.prange(tiles.shape[0]):
        tile: np.ndarray = tiles[i]
        meanColor: np.ndarray = np.array([
            np.mean(tile[:, :, 0]),
            np.mean(tile[:, :, 1]),
            np.mean(tile[:, :, 2])
            ])

        avgToTile[i] = meanColor
    return avgToTile, tiles


@numba.jit(nopython=True, parallel=True)
def roundColorToNearestAvailable(color: np.ndarray, availableColors: np.ndarray) -> int:
    distances = np.zeros(availableColors.shape[0])
    for i in nb.prange(availableColors.shape[0]):
        distances[i] = np.linalg.norm(availableColors[i] - color)

    return np.where(distances == min(distances))[0]


@numba.jit(nopython=True, parallel=True, nogil=True)
def CPU_compute(refImage: np.ndarray, outImage: np.ndarray, tileAvgVals: np.ndarray, tiles: np.ndarray,
                tileShape: tuple):
    for y, x in nb.pndindex(refImage.shape[:2]):
        refPixel = refImage[y, x]
        nx, ny = x * tileShape[1], y * tileShape[0]

        tileIndex = roundColorToNearestAvailable(refPixel, tileAvgVals)
        outImage[ny:ny + tileShape[0], nx:nx + tileShape[1]] = tiles[tileIndex][0]


@cuda.jit()
def GPU_compute(refImage: np.ndarray, outImage: np.ndarray, tileAvgVals: np.ndarray, tiles: np.ndarray,
                tileShape: tuple):
    x, y = cuda.grid(2)
    refPixel = refImage[y, x]
    nx, ny = x * tileShape[1], y * tileShape[0]

    tileIndex = 0
    distance = 1000
    for i in nb.prange(tileAvgVals.shape[0]):
        c = tileAvgVals[i]
        dist = math.sqrt(
            (c[0] - refPixel[0]) ** 2 + (c[1] - refPixel[1] ) ** 2 + (c[2] - refPixel[2]) ** 2)
        if dist < distance:
            distance = dist
            tileIndex = i

    c = outImage[ny:ny + tileShape[0], nx:nx + tileShape[1]]
    # print(tiles[tileIndex].shape[0],tiles[tileIndex].shape[1],tiles[tileIndex].shape[2],"\n",c.shape[0],c.shape[1],c.shape[2])
    t = tiles[tileIndex]

    for ty in range(t.shape[0]):
        for tx in range(t.shape[1]):
            for channel in range(3):
                outImage[ny + ty:ny + ty + 1, nx + tx:nx + tx + 1,channel] = t[ty, tx, channel]

def processImages(compute, refImage, outImage, avgVals,tiles,t):

    if compute == "numba-cpu":
        print("Beginning tiling...")
        CPU_compute(refImage, outImage, avgVals, tiles, t.shape)
        print("Finished.")

    elif compute == "numba-gpu":
        print("Warning: Only CUDA GPUs are supported. The CUDA toolkit must also be installed."
              " Get CUDA toolkit from: https://developer.nvidia.com/cuda-toolkit")

        threadsPerBlock = 32
        dimBlock = (threadsPerBlock, threadsPerBlock, 1)
        dimGrid = (math.ceil(refImage.shape[1] / threadsPerBlock), math.ceil(refImage.shape[0] / threadsPerBlock))

        GPU_compute[dimGrid, dimBlock](refImage, outImage, avgVals, tiles, t.shape)
    else:
        print("Error. Invalid compute option", compute)
        exit(-10)



def generate(refPath, savePath, tilesDir, compute: Literal["numba-cpu", "numba-gpu"] = "numba-cpu", downscale=True):
    """

    :param refPath: Image reference path
    :param savePath: Path to sace output to.
    :param tilesDir: images to use for tiling
    :param compute: compute option. Whether to use gpu or cpu for computation
    :param downscale: Whether to downscale reference image so that output image is about the same size
    :return:
    """

    rawtiles = loadTiles(tilesDir)
    avgVals, tiles = computeTileAverageValues(rawtiles)
    t = rawtiles[0]
    refImage: np.ndarray = cv2.imread(refPath)

    if downscale:
        refImage = cv2.resize(refImage, (round(refImage.shape[1] / t.shape[1]),
                                         round(refImage.shape[0] / t.shape[0]))
                              )
    outImage = np.zeros((refImage.shape[0] * t.shape[0], refImage.shape[1] * t.shape[1], 3))
    others.printImageOutputDetails(savePath, outImage.shape[1], outImage.shape[0])

    processImages(compute, refImage, outImage, avgVals, tiles, t)
    cv2.imwrite(savePath, outImage)
    print("Saved output to", savePath)
