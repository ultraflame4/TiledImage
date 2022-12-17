from pathlib import Path

import numpy as np
import typer
import numba
import enum

from PIL import Image

import TiledImage


class ProcessType(str, enum.Enum):
    njit = "njit",
    guvectorize = "guvectorize",
    cuda = "cuda"


def main(
        reference_imagepath: Path = typer.Argument(..., help="Path to reference image"),
        tileset_glob: str = typer.Argument(..., help="Glob Path to reference image. Eg: ./assets/tiles/*.png"),
        out_path: Path = typer.Argument(..., help="Path to save the final result to. ./out.png"),
        resize_factor: float = typer.Option("auto",
                                            help="Resize factor for reference image, so that the final image is not too big. Default: auto (resizes based on tile size)"),
        process_type: ProcessType = typer.Option(ProcessType.guvectorize,
                                                 help="Type of processing to use. Default: guvectorize. WARNING njit IS EXTREMELY SLOW"),
):
    tiles, tile_shape = TiledImage.load_imageset(Path(), tileset_glob)

    if isinstance(resize_factor, str):
        if resize_factor.lower() == "auto":
            resize_factor = 1 / max(tile_shape)
        else:
            typer.echo(f"Invalid resize_factor: {resize_factor}")
            return

    referenceImage = TiledImage.load_image(reference_imagepath, resize=resize_factor, silent=False)

    if process_type == ProcessType.njit:
        typer.echo(
            "Warning!!!. Using njit process type!!!!! This is EXTREMELY SLOW and should only be used for testing !!!")
        image = TiledImage.generate_tiledimage(referenceImage, tiles, tile_shape)
    elif process_type == ProcessType.cuda:
        typer.echo("Using cuda process type!!!!! This only works on CUDA enabled GPUS !!!")
        image = TiledImage.generate_tiledimage_gu(referenceImage, tiles, tile_shape, useCuda=True)
    else:
        typer.echo("Using default process type guvectorize...")
        image = TiledImage.generate_tiledimage_gu(referenceImage, tiles, tile_shape, useCuda=False)

    Image.fromarray(image).save(out_path)


if __name__ == "__main__":
    print("# TiledImage version:", TiledImage.__version__)
    typer.run(main)
