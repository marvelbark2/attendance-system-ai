#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    outp = r.recognize_google(audio)
    if (outp == 'can you hear me'):
        print("yeah !")
    else:
        print("nooh !")
# write audio to a RAW file
