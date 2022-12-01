import librosa
import soundfile
import speech_recognition as sr

r = sr.Recognizer()

x, sample_rate = librosa.load("../Vatterode_Kunstscheune_KaschubaSimple.wav")
sr_audio_file = sr.AudioFile("../Vatterode_Kunstscheune_KaschubaSimple.wav")

non_mute_sections = librosa.effects.split(x, top_db=30)
print(non_mute_sections)
section_start = None

min_section_length = 3


for i, non_mute_section in enumerate(non_mute_sections):
    if section_start is None:
        section_start = non_mute_section[0]
    section_end = non_mute_section[1]
    if section_end - section_start < min_section_length * sample_rate:
        continue

    soundfile.write(f"moments/moment{i}.wav", x[section_start: section_end], sample_rate)
    with sr_audio_file as source:
        audio = r.record(source, (section_end - section_start) / sample_rate, offset=section_start/sample_rate)

    with open(f"moments/moment{i}.txt", 'w') as f:
        f.write(r.recognize_google(audio, language="de"))
    section_start = None

# from pydub import AudioSegment
# from pydub.silence import split_on_silence
#
# start = time.time()
# song = AudioSegment.from_mp3("Wippra_Schieferhaus_Klaus_Feik.mp3")
# print(time.time() - start)
# start = time.time()
# for i, segment in enumerate(split_on_silence(song.get_sample_slice(0, song.frame_rate * 50), min_silence_len=500)):
#     segment.export(f"seg{i}.wav", format="wav")
