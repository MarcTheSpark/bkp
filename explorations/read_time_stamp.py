import random
import pandas as pd
import pathlib
import re


recordings_folder = pathlib.Path("InterviewRecordings")

transcriptions = {}
timestamps = {}

for file in recordings_folder.glob("*.transcription"):
    transcriptions[file.stem] = file.read_text().lower()

for file in recordings_folder.glob("*.timestamps"):
    df = pd.read_csv(file)

    max_char = max(df["char_position"])
    df = df.set_index("char_position").reindex(range(max_char + 1)).interpolate().fillna(0)
    df["audio_position"] = df["audio_position"].astype("int64")

    timestamps[file.stem] = df


word_counts = {}
for transcription_text in transcriptions.values():
    for word in transcription_text.replace("\n", " ").split(" "):
        word_counts[word] = word_counts.get(word, 0) + 1


def seek_phrase(phrase):
    words_in_transcript = [w for w in set(phrase.lower().split(" ")) if w in word_counts]
    if len(words_in_transcript) == 0:
        return None
    rarest_word = min(words_in_transcript, key=lambda word: word_counts[word])

    locations = tuple(
        (which_file, match.start())
        for which_file, file_text in transcriptions.items()
        for match in re.finditer(rarest_word, file_text)
    )

    which_file, char_position = random.choice(locations)
    return rarest_word, \
           str(recordings_folder.joinpath(which_file).with_suffix(".wav").resolve()), \
           timestamps[which_file].loc[char_position, "audio_position"]

# # print a list of words in order of how common
# print(sorted(list(word_counts.items()), key=lambda x: x[1]))
