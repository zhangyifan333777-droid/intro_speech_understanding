import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.
    
    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio
    
    @returns:
    text (str) - the recognized speech
    '''
    
    r = sr.Recognizer()
    
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    
    # Map simple language code to Google format
    if language == 'en':
        lang_code = 'en-US'
    elif language == 'ja':
        lang_code = 'ja-JP'
    else:
        lang_code = language   # allow custom language code
    
    text = r.recognize_google(audio, language=lang_code)
    
    return text
