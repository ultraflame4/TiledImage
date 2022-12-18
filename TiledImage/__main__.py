import os
from pathlib import Path

import typer

from PIL import Image
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.table import Table

import TiledImage
from TiledImage import video
from TiledImage.utils import ProcessType




def tiledImage_cli(
        reference_imagepath: Path = typer.Argument(..., help="Path to reference image"),
        out_path: Path = typer.Argument(..., help="Path to save the final result to. Eg ./out.png"),
        tileset_paths: list[Path] = typer.Argument(..., help="Path to images used as tiles. Eg: './assets/tiles/*.png' or './assets/tiles/a.png ./assets/tiles/n.png' ..."),
        resize_factor: float = typer.Option(-1, help="Resize factor for reference image, so that the final image is not too big. Default: -1 (resizes based on tile size)"),
        process_type: ProcessType = typer.Option(ProcessType.guvectorize, help="Type of processing to use. Default: guvectorize. WARNING njit IS EXTREMELY SLOW"),
):
    os.makedirs("./build/", exist_ok=True)

    overall_progress = TiledImage.utils.getProgressBar()
    overall_task = overall_progress.add_task("Overall Progress", total=5)

    with overall_progress:

        tiles, tile_shape = TiledImage.utils.load_imageset(Path(), "", tileset_paths, progress=overall_progress)
        overall_progress.advance(overall_task)

        if resize_factor < 0:
            if resize_factor == -1:
                resize_factor = 1 / max(tile_shape)
            else:
                typer.echo(f"Invalid resize_factor: {resize_factor}")
                return

        referenceImage = TiledImage.utils.load_image(reference_imagepath, resize=resize_factor, progress=overall_progress)
        overall_progress.advance(overall_task)

        if process_type == ProcessType.njit:
            overall_progress.print("Warning!!!. Using njit process type!!!!! This is EXTREMELY SLOW and should only be used for testing !!!")
            image = TiledImage.generate_tiledimage(referenceImage, tiles, tile_shape)
        elif process_type == ProcessType.cuda:
            overall_progress.print("Using cuda process type!!!!! This only works on CUDA enabled GPUS !!!")
            image = TiledImage.generate_tiledimage_gu(referenceImage, tiles, tile_shape, useCuda=True,progress=overall_progress)
        else:
            overall_progress.print("Using default process type guvectorize...")
            image = TiledImage.generate_tiledimage_gu(referenceImage, tiles, tile_shape, useCuda=False,progress=overall_progress)
        overall_progress.advance(overall_task)


        out_path.parent.mkdir(parents=True, exist_ok=True)
        overall_progress.update(overall_task,description="Overall Progress - Saving...", advance=1)
        Image.fromarray(image).save(out_path)
        overall_progress.print(f"Saved output to {out_path}")
        overall_progress.update(overall_task,description="Overall Progress - Finished", advance=1)


def main():
    app = typer.Typer()
    vidTyper = typer.Typer()
    print("# TiledImage version:", TiledImage.__version__)
    app.command(name="img",help="Generates a tiled image using a reference image and a set of images as tiles")(tiledImage_cli)

    app.command(name="vid",help="Generates a tiled image video by converting all of its frames into a tiled image.")(video.video_cli)

    app()

if __name__ == "__main__":
    main()
