import fire
from PIL import Image
import multiprocessing
import TiledImage as TM




def commandLine_generate(refPath,saveDir,tilesDir,downsize=True,keepRatio=True,quads=4,processes=1):
    max_processes=8
    processes=min(max_processes,processes)
    canvases_results = []

    t = TM.TiledImageMaker(TM.loadImagesFromFolder(tilesDir),Image.open(refPath))
    t.downsample=downsize
    t.keepRatio=keepRatio
    t.generate(quads,saveDir)

if __name__ == "__main__":

    fire.Fire(commandLine_generate)