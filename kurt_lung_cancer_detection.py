import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import pylab
import scipy.ndimage
import time
import matplotlib.pyplot as plt

DATA_PATH = '/Volumes/My Passport for Mac/stage1/'

# Timing function
def timeit(method):
    """
    Timing wrapper
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print 'Running %r took %2.4f sec\n' % \
              (method.__name__, te-ts)
        return result
    return timed

def show_slice(s):
    """
    Input:
        s -- dicom slice object
    Output:
        plots 2-D image of slice
    """
    pylab.imshow(s.pixel_array, cmap=pylab.cm.bone)
    pylab.show()

@timeit
# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    print len(slices)
    for s in slices:
        print s.SliceThickness
        print s.PixelSpacing
        print '\n'

if __name__ == "__main__":
    # load_scan(DATA_PATH + 'ff8599dd7c1139be3bad5a0351ab749a')
    load_scan(DATA_PATH + '0a0c32c9e08cc2ea76a71649de56be6d')
