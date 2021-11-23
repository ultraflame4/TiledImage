import TiledImage.__main__ as ti
import timeit

starttime = timeit.default_timer()
ti.commandLine_generate("./assets/blackhole1.jpg","./out1.png","./assets/tiles/*",compute_mode="old")
oldMode = timeit.default_timer() - starttime


starttime = timeit.default_timer()
ti.commandLine_generate("./assets/blackhole1.jpg","./out2.png","./assets/tiles/*")
newMode = timeit.default_timer() - starttime

print(f"\n-------------------------\nOldMode {oldMode}\n New mode {newMode}")
