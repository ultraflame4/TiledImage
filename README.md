# TiledImage

This program, using a reference image, tiles images provided to create a new image.

**Example:**

![tiledimage0](https://user-images.githubusercontent.com/34125174/147404436-d5be34b1-4816-4005-95b6-adffdaad8307.png)

Original image from https://wallpapercave.com/wp/wp2077533.jpg

### Install
Python required
Install with pip : `pip install git+https://github.com/ultraflame4/TiledImage.git`

### Usage
**Command**

`python -m TiledImage "./path/to/reference/image.png" "./savepath/out.png" "/path/tiles/folder/*" `

*Optional Flags*
- downsize : Scales down reference image, improves performances Default : True
- keepRatio : Maintain ration when scaling down Default : True
- quads : Number of threads . Default : 4
- scale : Custom scaling. Applied before downsizing. Default: 1
- compute_mode : Options on how to compute. Default: "old"
  - "old" : use the old way.
  - "numba-cpu": Uses numba jit and numpy on the cpu.
    - **Notes:**
      - quads and scale flags will be ignored
      
  - "numba-gpu": Uses numba cuda and numpy on the gpu.
    - **Notes:**
      - quads and scale flags will be ignored
      - Only supported on CUDA GPUs
      - CUDA toolkit must be installed
