import cv2
import dicom
from keras.models import Sequential
from keras.layers.pooling import MaxPooling3D
from keras.layers.convolutional import Convolution3D
from keras.layers.core import Dense, Dropout, Flatten
import numpy as np
import os
import pandas as pd
import pickle
import pylab
import scipy.ndimage
import time
import matplotlib.pyplot as plt

DATA_PATH = '/Volumes/My Passport for Mac/stage1/'

"""
Dataset:
1595 patients
1397 have ground truth

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
'PatientBirthDate' -- Jan. 1 1900 for all patients
'PatientID' -- same as folder name
'PatientName' -- meaningless string
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
        if s.shape[2] == 1:
            s = s.reshape((s.shape[0], s.shape[1]))
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

def BatchGenerator(batch_size = 4):
    # Get metadata
    patient_df = pd.read_csv('patient_metadata.csv')

    # Get training data
    training_patients = list(patient_df[np.isfinite(patient_df.cancer)].patient_id)

    # Make batches
    X_train_batch = []
    y_train_batch = []
    ctr = 0
    for patient in training_patients:
        path = DATA_PATH + patient
        if patient[0] != '.':
            if patient + '_resampled_pad.npy' in os.listdir(path + '/'):
                with open(path + '/' + patient + '_resampled_pad.npy', 'r') as f:
                    image = np.load(f)
                    image = image.reshape((530, 490, 490, 1))
                    print image.shape
                    X_train_batch.append(image)
                    label = int(patient_df[patient_df.patient_id == patient].cancer)
                    y_train_batch.append(label)
                ctr += 1
            if ctr == batch_size:
                ctr = 0
                yield np.stack(X_train_batch), np.array(y_train_batch)

def train_cnn():
    parameters = {
                    'n_epochs': 2,
                    'nb_filters': 16,
                    'kernal_dim1': 5,
                    'kernal_dim2': 5,
                    'kernal_dim3': 5,
                    'dim_ordering': 'tf',
                    'pool_dimensions': (2, 2, 2)
                 }

    model = Sequential()
    model.add(Convolution3D(parameters['nb_filters'],
                            parameters['kernal_dim1'],
                            parameters['kernal_dim2'],
                            parameters['kernal_dim3'],
                            input_shape = (530, 490, 490, 1),
                            dim_ordering = parameters['dim_ordering']))
    print "Dimensions after first convolution layer: {}".format(model.output_shape)
    model.add(MaxPooling3D(pool_size = parameters['pool_dimensions'],
                           dim_ordering = parameters['dim_ordering']))
    print "Dimensions after first pooling layer: {}".format(model.output_shape)
    model.add(Dropout(0.5))
    print "Dimensions after first dropout layer: {}".format(model.output_shape)
    model.add(Flatten())
    print "Dimensions after first flatten layer: {}".format(model.output_shape)
    """
    model.add(Flatten())
    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Activation('softmax'))
    model.compile(optimizer = 'sgd',
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    # Train in batches
    for epoch in xrange(parameters['n_epochs']):
        print "Epoch {}".format(epoch + 1)
        for X_train, Y_train in BatchGenerator():
            print X_train.shape
            print Y_train.shape
            model.fit(X_train, Y_train, batch_size=4, nb_epoch=1, verbose = 2)
    """

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
    # pad_dimensions()

    train_cnn()
