import TiledImage.__main__ as ti
import timeit

# starttime = timeit.default_timer()
# ti.commandLine_generate("./assets/blackhole1.jpg","./out1.png","./assets/tiles/*",compute_mode="old",downsize=False)
# oldMode = timeit.default_timer() - starttime
#

starttime = timeit.default_timer()
ti.commandLine_generate("./assets/ref.png","./out2.png","./assets/tiles/*" ,compute_mode="numba-cpu",downsize=True)
newMode = timeit.default_timer() - starttime

print(f"\n-------------------------\nOldMode {''}\n New mode {newMode}")
