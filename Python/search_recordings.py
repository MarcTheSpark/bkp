import logging
from huggingface_hub.inference_api import InferenceApi
import numpy as np
from pathlib import Path
import dataclasses
import yaml
import pathlib

with open("config.yaml", "r") as f:
    conf = yaml.safe_load(f)

inference = InferenceApi(
    "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
    token = "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",
    task = "feature-extraction"
    )


@dataclasses.dataclass
class TranscriptMatch:
    score: float
    file_path: str
    text: str
    recording_start_sample: int
    recording_end_sample: int


def _calculate_similarities(test_vec, vecs):
    return np.matmul(vecs, test_vec) / np.linalg.norm(vecs, axis=1) / np.linalg.norm(test_vec)


def _get_matches_for_recording(recording_path, input_embedding, how_many=5):
    rp = Path(recording_path)
    text_chunks_file = rp.with_suffix(".textchunks")
    embeddings_file = rp.with_suffix(".vectors.npy")
    audio_locations_file = rp.with_suffix(".samplocations.npy")

    if not all(x.exists() for x in (text_chunks_file, embeddings_file, audio_locations_file)):
        logging.info(f"Processing of {recording_path} incomplete. Skipping.")
        return ()

    snippet_text_chunks = text_chunks_file.read_text().strip().split("\n")
    snippet_embeddings = np.load(embeddings_file)
    snippet_audio_locations = np.load(audio_locations_file)

    similarities = _calculate_similarities(input_embedding, snippet_embeddings)
    top_indices = np.argsort(similarities)
    matches = []

    for i in top_indices[-how_many:]:
        matches.append(
            TranscriptMatch(similarities[i], str(rp.resolve()), snippet_text_chunks[i],
                            int(snippet_audio_locations[i][0]), int(snippet_audio_locations[i][1]))
        )
    return matches


def find_best_matches_in_all_recordings(text, how_many=3, max_from_one_source=2):
    input_vec = inference(
        inputs=[text]
    )[0]
    top_matches = []
    for path in conf["paths"]:
        for recording in pathlib.Path(path).glob("*.wav"):
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
        logging.getLogger().setLevel(logging.INFO)
        input_sentence = input("Enter a sentence: ")
        print(find_best_matches_in_all_recordings(input_sentence))
