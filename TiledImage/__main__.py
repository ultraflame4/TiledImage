import os
from pathlib import Path

import numpy as np
import typer
import numba
import enum

from PIL import Image
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table

import TiledImage
import TiledImage.utils


class ProcessType(str, enum.Enum):
    njit = "njit",
    guvectorize = "guvectorize",
    cuda = "cuda"


def tiledImage_cli(
        reference_imagepath: Path = typer.Argument(..., help="Path to reference image"),
        out_path: Path = typer.Argument(..., help="Path to save the final result to. ./out.png"),
        tileset_paths: list[Path] = typer.Argument(..., help="Path to images used as tiles. Eg: './assets/tiles/*.png' or './assets/tiles/a.png ./assets/tiles/n.png' ..."),
        resize_factor: float = typer.Option(-1,
                                            help="Resize factor for reference image, so that the final image is not too big. Default: -1 (resizes based on tile size)"),
        process_type: ProcessType = typer.Option(ProcessType.guvectorize,
                                                 help="Type of processing to use. Default: guvectorize. WARNING njit IS EXTREMELY SLOW"),
):
    os.makedirs("./build/", exist_ok=True)

    overall_progress = Progress("{task.description}",BarColumn(),TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
    overall_task = overall_progress.add_task("Generating Tiled Image...[Overall]", total=4)
    load_image_progress = Progress("{task.description}",BarColumn(),TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
    compute_progress = TiledImage.SpinnerProgress()
    progress_table = Table.grid()
    progress_table.add_row(load_image_progress)
    progress_table.add_row(compute_progress)
    progress_table.add_row(overall_progress)

    with Live(progress_table, refresh_per_second=10, ):

        tiles, tile_shape = TiledImage.utils.load_imageset(Path(), "", tileset_paths, progress=load_image_progress)
        overall_progress.advance(overall_task)

        if resize_factor < 0:
            if resize_factor == -1:
                resize_factor = 1 / max(tile_shape)
            else:
                typer.echo(f"Invalid resize_factor: {resize_factor}")
                return

        referenceImage = TiledImage.utils.load_image(reference_imagepath, resize=resize_factor, progress=load_image_progress)
        overall_progress.advance(overall_task)

        if process_type == ProcessType.njit:
            overall_progress.print("Warning!!!. Using njit process type!!!!! This is EXTREMELY SLOW and should only be used for testing !!!")
            image = TiledImage.generate_tiledimage(referenceImage, tiles, tile_shape)
        elif process_type == ProcessType.cuda:
            overall_progress.print("Using cuda process type!!!!! This only works on CUDA enabled GPUS !!!")
            image = TiledImage.generate_tiledimage_gu(referenceImage, tiles, tile_shape, useCuda=True,progress=compute_progress)
        else:
            overall_progress.print("Using default process type guvectorize...")
            image = TiledImage.generate_tiledimage_gu(referenceImage, tiles, tile_shape, useCuda=False,progress=compute_progress)
        overall_progress.advance(overall_task)


        out_path.parent.mkdir(parents=True, exist_ok=True)
        overall_progress.print(f"Saving result to {out_path}...")
        Image.fromarray(image).save(out_path)
        overall_progress.print(f"Saved")
        overall_progress.advance(overall_task)


def main():
    print("# TiledImage version:", TiledImage.__version__)

    typer.run(tiledImage_cli)

if __name__ == "__main__":
    main()
