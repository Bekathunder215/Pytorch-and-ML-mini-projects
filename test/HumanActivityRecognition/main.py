import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow


LOAD_TXT=0
#getting data ready from txt file
def LoadtxtToDF():
    processed_lines = []
    with open("data/WISDM_ar_v1.1_raw.txt","r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                line = line.split(";")
                if len(line) >2:
                    line[1] + "\n"
                    lines.append(line[0])
                    lines.append(line[1])
                    del lines[i]
            except:
                print("error at line")
                break
        for i in lines:
            processed_lines.append([item for item in i.strip(";\n").split(",")])

    for i, item in enumerate(processed_lines):
        if len(processed_lines[i])>6:
            processed_lines[i].pop()
    cols = ["user", "activity", "timestamp", "x-accel", "y-accel", "z-accel"]
    df = pd.DataFrame(data=processed_lines, columns=cols)
    return df

#saving and loading a custom csv to a pandas dataframe for manipulation
def savedftocsv(df):
    df.to_csv("data/ReadyDF.csv",index_label=False)
def loaddf():
    return pd.read_csv("data/ReadyDF.csv")

if LOAD_TXT:
    df = LoadtxtToDF()
    savedftocsv(df)
else:
    df = loaddf()

#reading csv is easier for machine
#check for na values and dropthem

#print(df.isna().sum())
df.dropna(inplace=True)
#print(df.isna().sum()) for doublechecking

#check for value counts on all activities
    #print(df["activity"].value_counts())
#we see that the output for standing is low and the second lowest is sitting
#we have to balance the data by choosing len(standing) from all the other ones in random
Fs = 20
activities = df["activity"].unique()

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15,7), sharex=True)
    plot_axis(ax0, data["timestamp"], data["x-accel"], "X-Axis")
    plot_axis(ax1, data["timestamp"], data["y-accel"], "Y-Axis")
    plot_axis(ax2, data["timestamp"], data["z-accel"], "Z-Axis")
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def plot_axis(ax,x,y,title):
    ax.plot(x,y,'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y)+np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def plot():
    for activity in activities:
        data_for_plot = df[(df["activity"] == activity)][:Fs*10]
        plot_activity(activity, data_for_plot)

data = df.drop(["user", "timestamp"], axis=1).copy()

label = LabelEncoder()
data['label'] = label.fit_transform(data["activity"])
#standardize data
X = data[["x-accel", "y-accel", "z-accel"]] #features
y = data["label"] #targets
scaler = StandardScaler()
X = scaler.fit_transform(X)
scaled_X=pd.DataFrame(data=X, columns=["x-accel", "y-accel", "z-accel"])
scaled_X["label"] = y.values

#frame preparation
frame_size = Fs*4 #80 samples  meaning the samples will be 80x3
hop_size = Fs*2 #hop size is the step on the data, choosing 2 is giving us an overlapping

def get_frames(df, frame_size, hop_size):

    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x-accel'].values[i: i + frame_size]
        y = df['y-accel'].values[i: i + frame_size]
        z = df['z-accel'].values[i: i + frame_size]
        
        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

X, y = get_frames(scaled_X, frame_size, hop_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
#we make it 3d data
X_train = X_train.reshape(21963, 80, 3, 1)
X_test = X_test.reshape(5491, 80, 3, 1)

model = tensorflow.keras.Sequential()
model.add(tensorflow.keras.layers.Conv2D(16, (2,2), activation = "relu", input_shape = X_train[0].shape))
model.add(tensorflow.keras.layers.Dropout(0.1))
model.add(tensorflow.keras.layers.Conv2D(32, (2,2), activation = "relu"))
model.add(tensorflow.keras.layers.Dropout(0.2))

model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(64, activation="relu"))
model.add(tensorflow.keras.layers.Dropout(0.5))

model.add(tensorflow.keras.layers.Dense(6, activation="softmax"))
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
print(history)