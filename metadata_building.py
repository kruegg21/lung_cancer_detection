import os
import pandas as pd

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
    df.to_csv('patient_metadata.csv', index = False)
    return df

def add_ground_truth():
    df = pd.read_csv('patient_metadata.csv')
    labels_df = pd.read_csv('stage1_labels.csv')

    # Merge DataFrames
    merged_df = df.merge(labels_df,
                         how = 'left',
                         left_on = 'patient_id',
                         right_on = 'id')
    merged_df.drop('id', axis = 1, inplace = True)
    merged_df.to_csv('patient_metadata.csv', index = False)

if __name__ == "__main__":
    add_ground_truth()
