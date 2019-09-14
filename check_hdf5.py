import h5py

#for filename in config.HDF5_OUTPUTS:
filename = "../clean_data/hdf5/train.hdf5"
db = h5py.File(filename, "r")
print(db["images"].shape)
db.close()
