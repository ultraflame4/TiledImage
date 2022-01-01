import TiledImage.__main__ as ti
import timeit

# starttime = timeit.default_timer()
# ti.commandLine_generate("./assets/blackhole1.jpg","./out1.png","./assets/tiles/*",compute_mode="old",downsize=False)
# oldMode = timeit.default_timer() - starttime
#

starttime = timeit.default_timer()
ti.commandLine_generate("./assets/blackhole1.jpg","./out2.png","./assets/tiles/*" ,compute_mode="numba-gpu",downsize=False)
newMode = timeit.default_timer() - starttime

print(f"\nTime taken {newMode}")
