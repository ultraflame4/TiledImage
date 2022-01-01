import sys
import time

import click
import TiledImage as TM



@click.command()
@click.argument("reference_path")
@click.argument("tilesdir")
@click.argument("output_path")
@click.option("-t","--timeit","timeit",default=False, help="If enabled, will print out total time takened")
@click.option("-m","--max-memory","max_memory",default=2048,help="The maximum amount of memory to use when generating the image."
                                                                 "This option does not dictate the total amount of memory the program will use."
                                                                 "Program will split and queue workload according to amount of memory available."
                                                                 "Lower memory will result in longer run times as there will be lesser amount of work that can be done in parallel")
def commandLine_generate(reference_path:str, tilesdir:str, output_path:str, timeit:bool,max_memory:int):
    """
    reference_path: Path to reference image. Reference image is what TiledImage attempts to create using the tiles\n
    tilesdir:  Path to directory containg the tile images. Tiles are images that are tiled together to create a image similar to the reference image.\n
    output_path: Path of the file to save resulting image to.
    """
    print(f"Tiled Image version: {TM.__version__}")
    # print(f"Command Arguments {reference_path} {tilesdir} {output_path} {timeit}")
    now = time.time()

    # code here

    if timeit:
        print(f"Time taken: {time.time()-now}ms")


if __name__ == "__main__":
    commandLine_generate()