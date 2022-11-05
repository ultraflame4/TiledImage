import numpy as np
import numba as nb


@nb.guvectorize([(nb.int64, nb.int64[:,:,:], nb.int64[:])], '(),(a,b,c  )->()')
def g(a, b, out):
    print(a,b)

# represent reference image
a=np.zeros((2,2,3),dtype=np.int64)

# represent precomputed tile values
b=np.ones((3,3,3),dtype=np.int64)


g(a, b)

print("__a__")
print(a)
print("__b__")
print(b)
