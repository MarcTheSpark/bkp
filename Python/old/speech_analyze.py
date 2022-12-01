import speech_recognition as sr

r = sr.Recognizer()
audio_file = sr.AudioFile("Vatterode_Kunstscheune_Kaschuba.wav")

with audio_file as source:
    r.adjust_for_ambient_noise(source)
    audio = r.record(source, 10)
    text = r.recognize_google(audio, language="de")
    print(text)


