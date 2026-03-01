import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    num_frames = 1 + (len(waveform) - frame_length) // step
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]
    return frames

def VAD(waveform, Fs):
    # Frame parameters
    frame_length = int(0.025*Fs)
    step = int(0.01*Fs)
    frames = waveform_to_frames(waveform, frame_length, step)
    energy = np.sum(frames**2, axis=1)
    threshold = 0.1 * np.max(energy)
    speech_flags = energy > threshold

    # Group consecutive speech frames into segments
    segments = []
    start_idx = None
    for i, flag in enumerate(speech_flags):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            end_idx = i
            segment = waveform[start_idx*step : start_idx*step + (end_idx-start_idx)*frame_length]
            segments.append(segment)
            start_idx = None
    if start_idx is not None:
        segment = waveform[start_idx*step : start_idx*step + (len(speech_flags)-start_idx)*frame_length]
        segments.append(segment)
    return segments

def segments_to_models(segments, Fs):
    models = []
    N = int(0.004*Fs)
    frame_len = N
    step = int(0.002*Fs)
    for seg in segments:
        # Pre-emphasis
        preemph = np.append(seg[0], seg[1:] - 0.97*seg[:-1])
        frames = waveform_to_frames(preemph, frame_len, step)
        spec = np.abs(np.fft.fft(frames, axis=1))
        low_half = spec[:, :frame_len//2]
        model = np.mean(low_half, axis=0)
        models.append(model)
    return models

def recognize_speech(testspeech, Fs, models, labels):
    # Chop test speech into frames
    test_segments = VAD(testspeech, Fs)
    sims = np.zeros((len(models), len(test_segments)))
    test_outputs = []
    for j, tseg in enumerate(test_segments):
        N = int(0.004*Fs)
        frame_len = N
        step = int(0.002*Fs)
        preemph = np.append(tseg[0], tseg[1:] - 0.97*tseg[:-1])
        frames = waveform_to_frames(preemph, frame_len, step)
        spec = np.abs(np.fft.fft(frames, axis=1))
        low_half = spec[:, :frame_len//2]
        test_model = np.mean(low_half, axis=0)
        # Cosine similarity
        cos_sims = []
        for model in models:
            sim = np.dot(model, test_model) / (np.linalg.norm(model)*np.linalg.norm(test_model)+1e-8)
            cos_sims.append(sim)
        sims[:, j] = cos_sims
        idx = np.argmax(cos_sims)
        test_outputs.append(labels[idx])
    return sims, test_outputs
