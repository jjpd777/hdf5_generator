import h5py
from utils import config as config

for filename in config.HDF5_OUTPUTS:
    db = h5py.File(filename, "r")
    print(db["images"].shape)
    db.close()
