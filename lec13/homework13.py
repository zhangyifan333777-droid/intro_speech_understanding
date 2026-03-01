import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):

    nframes = int((len(speech) - frame_length) / frame_skip)

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        a = librosa.lpc(frame, order)
        A[i, :] = a

        e = np.zeros(frame_length)
        for n in range(order, frame_length):
            e[n] = frame[n] + np.dot(a[1:], frame[n-order:n][::-1])

        excitation[i, :] = e

    return A, excitation


def synthesize(e, A, frame_skip):

    nframes = A.shape[0]
    order = A.shape[1] - 1

    synthesis = np.zeros(frame_skip * nframes)

    for i in range(nframes):
        start = i * frame_skip
        frame_e = e[start:start + frame_skip]
        a = A[i]

        y = np.zeros(frame_skip)

        for n in range(frame_skip):
            y[n] = frame_e[n]
            for k in range(1, min(order + 1, n + 1)):
                y[n] -= a[k] * y[n - k]

        synthesis[start:start + frame_skip] = y

    return synthesis


def robot_voice(excitation, T0, frame_skip):

    nframes = excitation.shape[0]

    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):
        gain[i] = np.sqrt(np.mean(excitation[i] ** 2))

        start = i * frame_skip

        for n in range(frame_skip):
            if n % T0 == 0:
                e_robot[start + n] = gain[i]

    return gain, e_robot
