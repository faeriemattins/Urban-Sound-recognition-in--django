import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd

from tensorflow.keras.models import model_from_json
from tensorflow import keras
import tensorflow as tf

classDict = {0: "Air Conditioner", 1: "Car Horn", 2: "Children Playing", 3: "Dog Bark", 4: "Drilling",
             5: "Engine Idling", 6: "Gun shot", 7: "Jack hammer", 8: "Siren", 9: "Street Music"}


def load_model():
    global model
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    # print("Loaded model from disk")


def predict(file_name):

    #file_name = "7061-6-0-0.wav"
    # Here kaiser_fast is a technique used for faster extraction
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    # We extract mfcc feature from data
    mels = np.mean(librosa.feature.melspectrogram(
        y=X, sr=sample_rate).T, axis=0)
    mels = mels.reshape(1, 16, 8, 1)
    a = model.predict(mels)
    classid = (np.argmax(a))
    className = classDict[classid]
    return className


def plot(file_name):
    plt.figure(figsize=(12, 4))
    data, sample_rate = librosa.load(file_name)
    _ = librosa.display.waveplot(data, sr=sample_rate)
    ipd.Audio(file_name)

    """
    dat1, sampling_rate1 = librosa.load(file_name)
    dat2, sampling_rate2 = librosa.load(file_name)
    plt.figure(figsize=(20, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    """

load_model()
print(predict("7061-6-0-0.wav"))
plot("7061-6-0-0.wav")