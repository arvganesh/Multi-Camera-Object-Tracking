import h5py
import numpy as np
import random
from pprint import pprint

# arr = random.sample(range(1,100), 10)

# with h5py.File('random.hdf5', 'w') as f:
#     dset = f.create_dataset("default", data=arr)
#     dset = f.create_dataset("yeet", data=arr[:5])
#     f.close()

# with h5py.File('random.hdf5', 'r') as f:
#     deff = f['default']
#     yeet = f['yeet']
#     print(deff, yeet)
#     f.close()


x = np.array([[0,0,0,1],
              [0,0,1,1],
              [0,1,0,0],
              [1,1,0,0]], 
              dtype=np.bool_)   
pprint(x)
x = np.delete(x, 2, axis=0)
x = np.delete(x, 2, axis=1)
pprint(x) 