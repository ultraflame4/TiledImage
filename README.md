# TilImg V3
This program composites many images into one big image using a set of images and a reference image
![results](https://user-images.githubusercontent.com/34125174/208235487-44f5e641-e6eb-453a-a9db-25d93a093782.png)
[Generated from Photo by Pixabay from Pexels: https://www.pexels.com/photo/dock-under-cloudy-sky-in-front-of-mountain-206359/ )

### Installation & Usage
Check the [releases](https://github.com/ultraflame4/TiledImage/releases) page for the latest version and instructions on how to install it.

### Image
Generates a tiled image from a set of images and a reference image.
Usage: `timg img [OPTIONS] REFERENCE_IMAGEPATH OUT_PATH`

Usage (Before v3.1.0): `timg REFERENCE_IMAGEPATH OUT_PATH`
#### Arguments / Parameters
##### Required Arguments
- reference_imagepath: The path to the reference image which is used to determine the look of the final image 
- out_path: Where to save the results
- tileset_path: The path to the folder containing the images to be used as tiles to construct the final image

##### Optional Arguments
- --resize-factor: The factor by which to resize the reference image. Default is -1 (auto, resizes based on tile size. Final image resolution will stay mostly the same)
- --process-type: TilImg uses numba to speed up computation. This argument specifies the method used to do so. Default is guvectorize
  - guvectorize: Uses numba's guvectorize to speed up computation. This is the default method
  - njit: Uses numba's njit to speed up computation. This is known to be extremely slow
  - cuda: Also uses numba's guvectorize but targets CUDA-enabled GPUs. This is known to be slighly faster than guvectorize but requires a CUDA-enabled GPU AND has some overhead costs
  
### Video
Available form v3.1.0
Generates a video from a set of images and a reference video. Basically the video version of the image command.
This requires **ffmpeg to be installed** and to be **in the PATH!**
If you don't have ffmpeg installed, you can download it [here](https://ffmpeg.org/download.html)

Usage: `timg vid [OPTIONS] SOURCE_PATH SAVE_PATH TILESET_PATHS...`

##### Required Arguments
- source_path: The path to the reference image which is used to determine the look of the final image 
- save_path: Where to save the results
- tileset_path: The path to the folder containing the images to be used as tiles to construct the final image

##### Optional Arguments
- --resize-factor: The factor by which to resize the reference image. Default is -1 (auto, resizes based on tile size. Final image resolution will stay mostly the same)
- --process-type: TilImg uses numba to speed up computation. This argument specifies the method used to do so. Default is guvectorize
  - guvectorize: Uses numba's guvectorize to speed up computation. This is the default method
  - ~~njit: Uses numba's njit to speed up computation. This is known to be extremely slow~~ Not available in video mode
  - cuda: Also uses numba's guvectorize but targets CUDA-enabled GPUs. This is known to be slighly faster than guvectorize but requires a CUDA-enabled GPU AND has some overhead costs

### Limitations
- All images in the set must be of the same resolution and dimensions
- The final product may have black bars on the side or have clipped textures because<br/>
  the dimensions of the reference image do not match dimensions of the images in the image set.<br/>
  The reference image dimensions has to be a multiple of the image set's dimensions to avoid this.<br/>
  
- For video, ffmpeg must be installed and in the PATH!!!