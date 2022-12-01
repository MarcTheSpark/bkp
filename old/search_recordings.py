from huggingface_hub.inference_api import InferenceApi
import numpy as np
import pandas as pd
from pathlib import Path
import dataclasses


inference = InferenceApi(
    "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
    token = "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",
    task = "feature-extraction"
    )


@dataclasses.dataclass
class TranscriptMatch:
    score: float
    file_path: str
    transcription_start_char: int
    transcription_end_char: int
    text: str
    recording_start_sample: int
    recording_end_sample: int


def _calculate_similarities(test_vec, vecs):
    return np.matmul(vecs, test_vec) / np.linalg.norm(vecs, axis=1) / np.linalg.norm(test_vec)


def _get_matches_for_recording(recording_path, input_embedding, how_many=5):
    rp = Path(recording_path)
    all_snippet_embeddings = np.load(rp.with_suffix(".vectors.npy"))
    snippet_character_locations = np.load(rp.with_suffix(".locations.npy"))
    time_stamps_df = pd.read_csv(rp.with_suffix(".timestamps"))

    similarities = _calculate_similarities(input_embedding, all_snippet_embeddings)
    top_indices = np.argsort(similarities)

    max_char = max(time_stamps_df["char_position"])
    time_stamps_df = time_stamps_df.set_index("char_position").reindex(range(max_char + 20000)).interpolate().fillna(0)
    time_stamps_df["audio_position"] = time_stamps_df["audio_position"].astype("int64")


    matches = []

    for i in top_indices[-how_many:]:
        start_char, end_char = snippet_character_locations[i]
        matches.append(
            TranscriptMatch(similarities[i], str(rp.resolve()),
                            start_char, end_char,
                            rp.with_suffix(".transcription").read_text()[start_char: end_char],
                            time_stamps_df.loc[start_char, "audio_position"],
                            time_stamps_df.loc[end_char, "audio_position"])
        )
    return matches


def find_best_matches_in_all_recordings(text, how_many=3, max_from_one_source=2):
    input_vec = inference(
        inputs=[text]
    )[0]
    top_matches = []
    for recording in Path("InterviewRecordings").glob("*.wav"):
        if recording.with_suffix(".locations.npy").exists() and recording.with_suffix(".timestamps").exists() \
                and recording.with_suffix(".transcription").exists() and recording.with_suffix(".vectors.npy").exists():
            top_matches.extend(_get_matches_for_recording(recording, input_vec, how_many=how_many))
    top_matches.sort(key=lambda tm: tm.score, reverse=True)
    source_count = {}
    to_return = []
    for match in top_matches:
        if len(to_return) >= how_many:
            break
        source_count[match.file_path] = source_count.get(match.file_path, 0) + 1
        if source_count[match.file_path] <= max_from_one_source:
            to_return.append(match)
    return to_return


if __name__ == "__main__":
    while True:
        input_sentence = input("Enter a sentence: ")
        print(find_best_matches_in_all_recordings(input_sentence))


# LOGIN:
# meredityman@gmail.com
# GA2o0Oow3UX3V3w^a@B6zKXe7i9qkxdC3s%T
