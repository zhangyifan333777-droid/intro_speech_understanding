import gtts
import speech_recognition as sr
import librosa
import soundfile as sf

def synthesize(text, lang, filename):

    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):

    r = sr.Recognizer()
    recognized_texts = []

    for text, lang, root in zip(texts, languages, filenames):

        mp3file = root + ".mp3"
        wavfile = root + ".wav"

        synthesize(text, lang, mp3file)

        y, sr_rate = librosa.load(mp3file, sr=None)
        sf.write(wavfile, y, sr_rate)

        with sr.AudioFile(wavfile) as source:
            audio = r.record(source)

        if lang == 'en':
            code = 'en-US'
        elif lang == 'ja':
            code = 'ja-JP'
        else:
            code = lang

        try:
            result = r.recognize_google(audio, language=code)
        except:
            result = ""

        recognized_texts.append(result)

    return recognized_texts
