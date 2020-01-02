cachedir = "cache"

from joblib import Memory
import time

memory = Memory(cachedir, verbose=0)

@memory.cache
def f(x):
    time.sleep(5)
    return x*2



print(f(100))
