import cv2
import numpy as np
import pandas as pd
import dicom
import os
import pickle
import pylab
import scipy.ndimage
import time
import matplotlib.pyplot as plt

DATA_PATH = '/Volumes/My Passport for Mac/stage1/'

"""
Dataset:
1595 patients

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
    try:
        pylab.imshow(s.pixel_array, cmap=pylab.cm.bone)
    except:
        pylab.imshow(s, cmap=pylab.cm.autumn)

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
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

@timeit
# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if s[-3:] == 'dcm']
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - \
                                 slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

@timeit
def resample_images(dump = True):
    resampled_dimensions = []
    for patient in os.listdir(DATA_PATH):
        path = path = DATA_PATH + patient
        if patient[0] != '.':
            if not any('npy' in item for item in os.listdir(path)):
                slices = load_scan(DATA_PATH + patient)
                image = stack_slices(slices)
                image, new_spacing = resample(image, slices, new_spacing=[1,1,1])
                print image.shape
                resampled_dimensions.append(image.shape)
                if dump:
                    with open(path + '/' + patient + '_resampled.npy', 'w+') as f:
                        np.save(f, image)

def BatchGenerator(batch_size = 16):
    batch = []
    ctr = 0
    for patient in os.listdir(DATA_PATH):
        path = path = DATA_PATH + patient
        if patient[0] != '.':
            print path + '/' + patient + '_resampled.npy'
            with open(path + '/' + patient + '_resampled.npy', 'r') as f:
                image = np.load(f)
                batch.append(image)
            ctr += 1
        if ctr == batch_size:
            ctr = 0
            yield np.stack(batch)

def find_dimensions():
    patient_id = []
    z = []
    y = []
    x = []
    for patient in os.listdir(DATA_PATH):
        path = path = DATA_PATH + patient
        if patient[0] != '.':
            with open(path + '/' + patient + '_resampled.npy', 'r') as f:
                image = np.load(f)
                patient_id.append(patient)
                z.append(image.shape[0])
                y.append(image.shape[1])
                x.append(image.shape[2])
    df = pd.DataFrame({'patient_id': patient_id, 'z': z, 'y': y, 'x': x})
    return df

def pad_dimensions():
    patient_metadata = pd.read_csv('patient_metadata.csv')

    longest_z_dimension = max(patient_metadata.z)
    longest_x_dimension = max(patient_metadata.x)
    longest_y_dimension = max(patient_metadata.y)

    for patient in os.listdir(DATA_PATH):
        path = path = DATA_PATH + patient
        if patient[0] != '.':
            with open(path + '/' + patient + '_resampled.npy', 'r') as f:
                image = np.load(f)

                # Determine how much to pad
                z_pad = longest_z_dimension - image.shape[0]
                x_pad = longest_x_dimension - image.shape[2]
                y_pad = longest_y_dimension - image.shape[1]
                padding = ((z_pad / 2, z_pad - z_pad / 2),
                           (y_pad / 2, y_pad - y_pad / 2),
                           (x_pad / 2, x_pad - x_pad / 2))

                # Create padded image
                image2 = np.pad(image,
                                pad_width = padding,
                                mode = 'constant',
                                constant_values = 0)

                # Save padded image to disk
                with open(path + '/' + patient + '_resampled_pad.npy', 'w+') as f:
                    np.save(f, image2)

# def train_cnn():
#     #
#     # model = Sequential()
#     # model.add(Convolution3D(
#
#     # Train in batches
#     for e in range(nb_epoch):
#     print("epoch %d" % e)
#     for X_train, Y_train in BatchGenerator():
#         model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)

if __name__ == "__main__":
    # Basic tests
    # patient = '0a0c32c9e08cc2ea76a71649de56be6d'
    # patient = 'ff8599dd7c1139be3bad5a0351ab749a'
    # slices = load_scan(DATA_PATH + patient)

    # EDA
    # number_of_slices = []
    # slice_thicknesses = []
    # pixel_spacing = []
    #
    # ctr = 0
    # number_patients = len(os.listdir(DATA_PATH))
    # for patient in os.listdir(DATA_PATH):
    #     if patient[0] != '.':
    #         slices = load_scan(DATA_PATH + patient)
    #         slice_thicknesses.append(slices[0].SliceThickness)
    #         number_of_slices.append(len(slices))
    #         pixel_spacing.append(slices[0].PixelSpacing)
    #     print slice_thicknesses
    #     print float(ctr) / number_patients
    #     ctr += 1

    # Resample images
    # resample_images()

    # Test
    pad_dimensions()
