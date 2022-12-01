import logging
import librosa
import speech_recognition as sr


INPUT_FILE = "../Vatterode_Kunstscheune_KaschubaSimple.wav"
OUTPUT_TRANSCRIPTION = "transcript/Transcription.txt"
OUTPUT_TIME_STAMPS = "transcript/TimeStamps.txt"

r = sr.Recognizer()

logging.getLogger().setLevel(logging.INFO)
logging.info("Loading audio file...")
x, sample_rate = librosa.load(INPUT_FILE, sr=None)
sr_audio_file = sr.AudioFile(INPUT_FILE)
logging.info("done.")

logging.info("Splitting audio into sections...")
non_mute_sections = librosa.effects.split(x, top_db=40)
logging.info(f"Done splitting audio into {len(non_mute_sections)} sections.")

section_start = None

min_section_length = 3

full_text = ""
time_stamps = "char_position,audio_position\n"

for i, non_mute_section in enumerate(non_mute_sections):
    if i % 10 == 0:
        logging.info(f"Analyzing sections {i}-{i + 9}")
    if section_start is None:
        section_start = non_mute_section[0]
    section_end = non_mute_section[1]
    if section_end - section_start < min_section_length * sample_rate:
        continue

    with sr_audio_file as source:
        audio = r.record(source, (section_end - section_start) / sample_rate, offset=section_start/sample_rate)

    recognized_audio = r.recognize_google(audio, language="de")

    full_text += " "
    time_stamps += f"\n{len(full_text)},{section_start}"
    full_text += recognized_audio

    section_start = None

with open(OUTPUT_TRANSCRIPTION, 'w') as f:
    f.write(full_text)

with open(OUTPUT_TIME_STAMPS, 'w') as f:
    f.write(time_stamps)
