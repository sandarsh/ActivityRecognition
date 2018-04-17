import os
import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
from scipy import signal
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Folder that contains data
data_folder = 'Data/'
output_folder = 'Features_1p5sec/'
# output_folder = 'Test/'

# Dataframe labels to column names
x = 1
y = 2
z = 3
label = 4

'''
0 - Unclassified
1 - Working at computer
2 - Climbing or coming down the stairs
3 - Standing
4 - Walking
'''

# Frequency of the sensor in use
sensor_frequency = 52

# Cuttoff frequency for low pass filter in hertz. Used in butterworth filter
cut_off_for_low_pass = 6

# Nyquist frequency
nyqHZ = sensor_frequency/2

# Number of seconds to use as window
window_size = 1.5

# window overlap percentage
overlap_pc = 10

# Number of samples in window
number_of_samples = int(sensor_frequency * window_size)


def get_current_window(index, file):
    curr_filepath = data_folder + file
    df = pd.read_csv(
        filepath_or_buffer=curr_filepath,
        skiprows=index,
        nrows=number_of_samples,
        header=None)
    return df


def zero_crossings(df, relevant_features):
    ans_arr = np.array([])
    for feature in relevant_features:
        curr_col = df[feature].values
        zc_num = ((curr_col[:-1] * curr_col[1:]) < 0).sum()
        ans_arr = np.append(ans_arr,zc_num)
    # print(ans_arr)
    return ans_arr


def p2p_val(df, relevant_features):
    ans_arr = np.array([])
    for feature in relevant_features:
        curr_col = df[feature].values
        curr_p2p = max(curr_col) - min(curr_col)
        ans_arr = np.append(ans_arr, curr_p2p)
    return ans_arr


def rms(df, relevant_features):
    ans_arr = np.array([])
    for feature in relevant_features:
        curr_col = df[feature].values
        curr_rms = np.sqrt(np.mean(curr_col**2))
        ans_arr = np.append(ans_arr, curr_rms)
    # print(ans_arr)
    return ans_arr


def kurtosis(df, relevant_features):
    ans_arr = np.array([])
    for feature in relevant_features:
        curr_col = df[feature].values
        curr_kurt = stats.kurtosis(curr_col, fisher=True, bias=False)
        ans_arr = np.append(ans_arr, curr_kurt)
    # print(ans_arr)
    return ans_arr


def skew(df, relevant_features):
    ans_arr = np.array([])
    for feature in relevant_features:
        curr_col = df[feature].values
        curr_skew = stats.skew(curr_col, bias=False)
        ans_arr = np.append(ans_arr, curr_skew)
    # print(ans_arr)
    return ans_arr


def crest_factor(df, relevant_features, rms_arr):
    ans_arr = np.array([])
    count = 0
    for feature in relevant_features:
        curr_col = df[feature].values
        curr_cf = (max(curr_col) / rms_arr[count])
        ans_arr = np.append(ans_arr, curr_cf)
        count += 1
    # print(ans_arr)
    return ans_arr


def vrms(df, relevant_features, initial_vel):
    ans_arr = np.array([])
    time = 1 / sensor_frequency
    for feature in relevant_features:
        vel_arr = np.array([])
        curr_col = df[feature].values
        for acc in curr_col:
            curr_vel = initial_vel + acc * time
            vel_arr = np.append(vel_arr, curr_vel)
            initial_vel = curr_vel
        curr_rms = np.sqrt(np.mean(vel_arr ** 2))
        ans_arr = np.append(ans_arr, curr_rms)
    # print(ans_arr)
    return ans_arr, initial_vel


def entropy(df, relevant_features):
    ans_arr = np.array([])
    for feature in relevant_features:
        # print(df[feature])
        binned = pd.cut(df[feature], bins=10, retbins=False, labels=False)
        # print(binned)
        p_data = binned.value_counts() / len(binned.values)
        # print("Yo",p_data)
        curr_ent = stats.entropy(p_data, qk=None, base=None)
        ans_arr = np.append(ans_arr, curr_ent)
        # print(curr_ent)
    # print(ans_arr)
    return ans_arr


def get_feature_list():
    relevant_features = ['lax', 'lay', 'laz',
                         'hax', 'hay', 'haz',
                         'lmag', 'hmag',
                         'lpc1', 'lpc2', 'lpc3',
                         'hpc1', 'hpc2', 'hpc3']

    feature_name = ['zc', 'p2p', 'rms', 'kurt', 'skew', 'cf', 'vrms', 'ent']
    feature_list = []
    for ftr in feature_name:
        for n in relevant_features:
            feature_list.append(n + "." + ftr)
    return feature_list


def write_features_to_file(f, vector):
    f.write(",".join(map(str,vector)))
    f.write("\n")


def aggregate_labels(curr_label):
    if curr_label == 5:
        curr_label = 2
    if curr_label == 7:
        curr_label = 3
    if curr_label == 6:
        curr_label = 4
    return curr_label


def get_window_label(df):
    df[label] = df[label].apply(aggregate_labels)
    return df[label].value_counts().idxmax()


def extract_features(df, initial_velocity, filename):
    features = [x, y, z]

    # -------------------------------------------------------------------
    # PCA Calculation
    # -------------------------------------------------------------------
    pcx = df.loc[:, features].values
    pcx = StandardScaler().fit_transform(pcx)
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(pcx)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['principal component 1',
                                         'principal component 2',
                                         'principal component 3'])
    df['pc1'] = principal_df['principal component 1']
    df['pc2'] = principal_df['principal component 2']
    df['pc3'] = principal_df['principal component 3']
    # -------------------------------------------------------------------
    df = df.drop([0], axis=1)   # Drop the inbuilt index column
    df['mag'] = np.sqrt(df[x]**2 + df[y]**2 + df[z]**2)    # Add magnitude

    # -------------------------------------------------------------------
    # Butterworth filter
    # -------------------------------------------------------------------
    b, a = signal.butter(9, cut_off_for_low_pass/nyqHZ, 'lowpass', False, 'ba')
    df['lax'] = signal.filtfilt(b, a, df[x])
    df['hax'] = df[x] - df['lax']
    df['lay'] = signal.filtfilt(b, a, df[y])
    df['hay'] = df[y] - df['lay']
    df['laz'] = signal.filtfilt(b, a, df[z])
    df['haz'] = df[z] - df['laz']
    df['lmag'] = signal.filtfilt(b, a, df['mag'])
    df['hmag'] = df['mag'] - df['lmag']
    df['lpc1'] = signal.filtfilt(b, a, df['pc1'])
    df['hpc1'] = df['pc1'] - df['lpc1']
    df['lpc2'] = signal.filtfilt(b, a, df['pc2'])
    df['hpc2'] = df['pc2'] - df['lpc2']
    df['lpc3'] = signal.filtfilt(b, a, df['pc3'])
    df['hpc3'] = df['pc3'] - df['lpc3']
    # -------------------------------------------------------------------

    # --------------------------------------------------------------------
    # Center the columns
    # --------------------------------------------------------------------
    df = df.drop([4], axis=1)  # Drop the label column. Labels stored in 'labels' variable.
    df = df - df.mean()
    # --------------------------------------------------------------------
    relevant_features = ['lax', 'lay', 'laz',
                         'hax', 'hay', 'haz',
                         'lmag', 'hmag',
                         'lpc1', 'lpc2', 'lpc3',
                         'hpc1', 'hpc2', 'hpc3']

    # feature_name = ['zc', 'p2p', 'rms', 'kurt', 'skew', 'cf', 'vrms', 'ent']
    zc_arr = zero_crossings(df, relevant_features)
    p2p_arr = p2p_val(df, relevant_features)
    rms_arr = rms(df, relevant_features)
    kurt_arr = kurtosis(df, relevant_features)
    skew_arr = skew(df, relevant_features)
    cf_arr = crest_factor(df, relevant_features, rms_arr)
    vrms_arr, initial_velocity = vrms(df, relevant_features, initial_velocity)
    ent_arr = entropy(df, relevant_features)
    ans = np.append(zc_arr,[p2p_arr,rms_arr,kurt_arr,skew_arr,cf_arr,vrms_arr,ent_arr])
    # print(len(ans),ans)
    # Printing and plotting code
    # print(df)
    # df.plot(y=['pc1', 'lpc1'])
    # pl.show()
    return ans, initial_velocity


def extract_feature_main():
    feature_list = get_feature_list()
    feature_list.append('label')
    # print(len(feature_list))

    # Traversing the data directory
    for dir, file, files in os.walk(data_folder):
        # Traversing each file
        for file in files:
            # check if its a csv file
            if '.csv' in file:
                output_file = output_folder + file[:-4] + "_ws_" + str(window_size) + "_opc_" + str(
                    overlap_pc) + ".features" + ".csv"
                with open(output_file, "w+") as f:
                    f.write(",".join(feature_list))
                    f.write("\n")
                    index = 0  # Tracking the index inside the file to load the window into the dataframe.
                    initial_velocity = 0
                    print("Reading from file", file)
                    while True:
                        df_current_window = get_current_window(index, file)
                        rows = df_current_window.shape[0]
                        cols = df_current_window.shape[1]
                        if rows < number_of_samples:
                            # print(rows)
                            break
                        index += int((number_of_samples - ((overlap_pc / 100) * number_of_samples)))
                        feature_vector, initial_velocity = extract_features(df_current_window, initial_velocity, file)
                        curr_label = get_window_label(df_current_window)
                        feature_vector = np.append(feature_vector, curr_label)
                        write_features_to_file(f, feature_vector)
                        # break
                f.close()
            # break
        # break


if __name__ == "__main__":
    extract_feature_main()
