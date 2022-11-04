import numpy as np
import numba as nb


@nb.guvectorize([(nb.int64[:], nb.int64[:,:,:], nb.int64[:])], '(n),(a,b,c)->(n)')
def g(a, b, out):
    print("-a")
    print(a)
    print("-b")
    print(b)

a=np.zeros((1,3,3),dtype=np.int64)
b=np.ones((4,4,3),dtype=np.int64)
g(a, b)

print("__a__")
print(a)
print("__b__")
print(b)
