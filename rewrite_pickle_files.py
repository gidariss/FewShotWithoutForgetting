import pickle
import glob
from time import time


def load_data(file_name):
    with open(file_name, "rb") as fo:
        data = pickle.load(fo, encoding='iso-8859-1')
    return data

fnames = glob.glob('datasets/MiniImagenet/*')

for fname in fnames:
    start = time()
    data = load_data(fname)
    r = time() - start
    print('old load time: ', fname, ': ', r)

    file_name = open(fname, 'wb')
    pickle.dump(data, file_name, protocol=-1)
    file_name.close()

    start = time()
    file_name = open(fname, 'rb')
    data_ = pickle.load(file_name)
    file_name.close()
    r = time() - start
    print('updated load time: ', fname, ': ', r)
