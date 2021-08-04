# TiledImage

This program, using a reference image, tiles images provided to create a new image.

### Install
Python required
Install with pip : `pip install git+https://github.com/ultraflame4/TiledImage.git`

### Usage
* Command *

`python -m TiledImage "./path/to/reference/image.png" "./savepath/out.png" "/path/tiles/folder/*" `

*Optional Flags*
- downsize : Scales down reference image, improves performances Default : True
- keepRatio : Maintain ration when scaling down Default : True
- quads : Number of processes/threads . Default : 4
