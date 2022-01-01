import sys
import time
from typing import Literal

import click
import TiledImage as TM
from TiledImage import core


@click.command()
@click.argument("reference_path")
@click.argument("tilesdir")
@click.argument("output_path")
@click.option("-t", "--timeit", "timeit", default=False,is_flag=True, help="If included, will print out total time takened")
@click.option("-m", "--max-memory", "max_memory", default=2048,
              help="units is in megabytes"
                   "The maximum amount of memory to use when generating the image."
                   "This option does not dictate the total amount of memory the program will use."
                   "Your computer RAM must still be able to handle the full size of resulting image + reference image + other stuff."
                   "Program will split and queue workload according to amount of memory available."
                   "Lower memory will result in longer run times as there will be lesser amount of work that can be done in parallel")
@click.option("--scale-down/--no-scale", "scale_down", default=True,
              help="Scales down reference image so that output image is about the same size as the original reference image."
                   "Warning: Setting this to False will result in higher memory comsumption. The max memory may need to be increased.")
@click.option("-c", "--compute-mode", "compute", default="cpu",
              help="Sets how the computation is done.                                                 "
                   "Options: 'normal','cpu','cuda'.                                                   "
                   "normal -> computation is done as normal. no speedups.                             "
                   "                                                     "
                   "cpu -> computation is accelerated with numba.                                     "
                   "cuda -> computation is accelerated with cuda and numba on the cuda supported gpus.")
@click.option("--verbose","verbose",default=False,is_flag=True)
def commandLine_generate(reference_path: str, tilesdir: str, output_path: str, timeit: bool, max_memory: int,
                         scale_down: bool, compute: core.c_option,verbose:bool):
    """
    reference_path: Path to reference image. Reference image is what TiledImage attempts to create using the tiles\n
    tilesdir:  Path to directory containg the tile images. Tiles are images that are tiled together to create a image similar to the reference image.\n
    output_path: Path of the file to save resulting image to.
    """
    print(f"Tiled Image version: {TM.__version__}")
    # print(f"Command Arguments {reference_path} {tilesdir} {output_path} {timeit}")
    now = time.time()

    # code here
    TM.generate(reference_path, tilesdir, output_path, timeit, max_memory, scale_down, compute,verbose)

    if timeit:
        print(f"Total Time taken: {time.time() - now}ms")


if __name__ == "__main__":
    commandLine_generate()
