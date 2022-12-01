import logging
import librosa
import speech_recognition as sr
import pathlib


def transcribe_audio(input_file):
    input_path = pathlib.Path(input_file)
    output_transcription = input_path.with_suffix(".transcription")
    output_time_stamps = input_path.with_suffix(".timestamps")

    if output_transcription.exists() and output_time_stamps.exists():
        logging.info(f"Recording {input_path.name} already analyzed. Skipping.")
        return

    r = sr.Recognizer()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"Loading audio file {input_file}...")
    x, sample_rate = librosa.load(str(input_path), sr=None)
    sr_audio_file = sr.AudioFile(str(input_path))
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
            audio = r.record(source, (section_end - section_start) / sample_rate, offset=section_start / sample_rate)

        recognized_audio = r.recognize_google(audio, language="de")

        time_stamps += f"\n{len(full_text)},{section_start}"
        full_text += recognized_audio + "\n"

        section_start = None

    logging.info("Saving data...")
    with open(output_transcription, 'w') as f:
        f.write(full_text)

    with open(output_time_stamps, 'w') as f:
        f.write(time_stamps)
    logging.info("Done.")


for file in pathlib.Path("InterviewRecordings").glob("*.wav"):
    transcribe_audio(file)
