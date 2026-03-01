import numpy as np
import torch
import torch.nn as nn
import librosa

def waveform_to_frames(waveform, frame_length, step):
    num_frames = 1 + (len(waveform) - frame_length) // step
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]
    return frames

def get_features(waveform, Fs):
    preemph = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    frame_len = int(0.004 * Fs)
    step = int(0.002 * Fs)
    frames = waveform_to_frames(preemph, frame_len, step)
    spec = np.abs(np.fft.fft(frames, axis=1))
    features = spec[:, :frame_len//2]

    vad_frame_len = int(0.025*Fs)
    vad_step = int(0.01*Fs)
    vad_frames = waveform_to_frames(waveform, vad_frame_len, vad_step)
    energy = np.sum(vad_frames**2, axis=1)
    threshold = 0.1 * np.max(energy)
    speech_flags = energy > threshold

    labels = []
    label_id = 0
    for i, flag in enumerate(speech_flags):
        if flag:
            labels.extend([label_id]*5)
            label_id += 1
    labels = np.array(labels[:features.shape[0]])

    return features, labels

def train_neuralnet(features, labels, iterations):
    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    NFEATS = features.shape[1]
    NLABELS = int(np.max(labels)) + 1

    model = nn.Sequential(
        nn.LayerNorm(NFEATS),
        nn.Linear(NFEATS, NLABELS)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    lossvalues = []
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(features_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        lossvalues.append(loss.item())

    return model, np.array(lossvalues)

def test_neuralnet(model, features):
    features_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = nn.functional.softmax(outputs, dim=1).detach().numpy()
    return probabilities
