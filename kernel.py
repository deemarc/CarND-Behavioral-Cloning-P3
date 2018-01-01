import csv
import cv2
import numpy as np
lines = []
#get fie description
with open('./data/driving_log.csv') as csvFile:
    #passing file description to csv fie reader
    reader = csv.reader(csvFile)
    #passing all csv info into a list
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
Y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x/255.0,input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,Y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')