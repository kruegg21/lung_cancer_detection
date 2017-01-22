import cv2
import numpy as np
import pandas as pd
import dicom
import os
import pylab
import scipy.ndimage
import time
import matplotlib.pyplot as plt

DATA_PATH = '/Volumes/My Passport for Mac/stage1/'

"""
Dataset:
1596 patients

Each patient is composed of a number of slices. Each slice has the following
attributes:
'AcquisitionNumber'
'BitsAllocated'
'BitsStored'
'Columns'
'FrameOfReferenceUID'
'HighBit'
'ImageOrientationPatient'
'ImagePositionPatient'
'InstanceNumber'
'KVP'
'Laterality'
'LongitudinalTemporalInformationModified'
'Modality'
'PatientBirthDate'
'PatientID'
'PatientName'
'PhotometricInterpretation'
'PixelData'
'PixelRepresentation'
'PixelSpacing'
'PositionReferenceIndicator'
'PregnancyStatus'
'RescaleIntercept'
'RescaleSlope'
'Rows'
'SOPClassUID'
'SOPInstanceUID'
'SamplesPerPixel'
'SeriesDescription'
'SeriesInstanceUID'
'SeriesNumber'
'SliceLocation'
'SliceThickness' -- distance between slices in millimeters
'SpecificCharacterSet'
'StudyInstanceUID'
'WindowCenter'
'WindowWidth'
'pixel_array'
"""

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

def show_slice(s, hist_equalize = False):
    """
    Input:
        s -- dicom slice object
    Output:
        Plots 2-D image of slice. The top image is the standard image, the
        bottom is histogram equalized (increases contrast)
    """
    pylab.subplot(2, 1, 1)
    pylab.imshow(s.pixel_array, cmap=pylab.cm.bone)

    if hist_equalize:
        pylab.subplot(2, 1, 2)
        pylab.imshow(cv2.equalizeHist(s.pixel_array.astype(np.uint8)),
                     cmap=pylab.cm.bone)
    pylab.show()

@timeit
def stack_slices(slices):
    """
    Input:
        slices -- list of slice objects
    Output:
        3-dimensional numpy array
    """
    image_3d = np.stack([s.pixel_array for s in slices])
    return image_3d

@timeit
# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - \
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
        print s.PixelSpacing

    print slices[0]
    for i in dir(slices[0]):
        print "\'{}\'".format(i)
    return slices

if __name__ == "__main__":
    # patient = '0a0c32c9e08cc2ea76a71649de56be6d'
    # patient = 'ff8599dd7c1139be3bad5a0351ab749a'
    # stack_slices(load_scan(DATA_PATH + patient))

    number_of_slices = []
    slice_thicknesses = []
    pixel_spacing = []

    ctr = 0
    number_patients = len(os.listdir(DATA_PATH))
    for patient in os.listdir(DATA_PATH):
        if patient[0] != '.':
            slices = load_scan(DATA_PATH + patient)
            slice_thicknesses.append(slices[0].SliceThickness)
            number_of_slices.append(len(slices))
            pixel_spacing.append(slices[0].PixelSpacing)
        print slice_thicknesses
        print float(ctr) / number_patients
