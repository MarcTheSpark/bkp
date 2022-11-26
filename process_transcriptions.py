
import re
from huggingface_hub.inference_api import InferenceApi
import numpy as np
import pathlib

inference = InferenceApi(
    "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
    token = "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",
    task = "feature-extraction"
    )


def process_transcription(file_path, sentence_length=12, skip=5, batch_size=100):
    path = pathlib.Path(file_path)
    locations_output_path = path.parent.joinpath(path.stem + ".locations.npy")
    vectors_output_path = path.parent.joinpath(path.stem + ".vectors.npy")
    if locations_output_path.exists() and vectors_output_path.exists():
        return

    with open(file_path) as f:
        text = f.read().replace("\n", " ")

    word_divisions = [0] + [x.start() for x in re.finditer(r"\s", text)]

    locations = []
    sentences = []
    for first_word_index in range(0, len(word_divisions) - sentence_length, skip):
        last_word_index = first_word_index + sentence_length
        locations.append((word_divisions[first_word_index], word_divisions[last_word_index]))
        sentences.append(text[word_divisions[first_word_index]: word_divisions[last_word_index]])

    for sentence_num in range(0, len(sentences), batch_size):
        partial_output_path = path.parent.joinpath(path.stem + f"-partial-{sentence_num:06d}.npy")
        if partial_output_path.exists():
            continue
        print(f"Processing segments {sentence_num}-{sentence_num + batch_size - 1} of file {path.name}...")
        this_batch_encodings = np.array(inference(inputs=sentences[sentence_num: sentence_num + batch_size]))

        np.save(partial_output_path, this_batch_encodings)
    combined_numpy = np.concatenate([np.array(np.load(str(glob)))
                                     for glob in sorted(path.parent.glob(f"{path.stem}-partial-*.npy"))])

    np.save(locations_output_path, np.array(locations))
    np.save(vectors_output_path, combined_numpy)
    for glob in path.parent.glob(f"{path.stem}-partial-*.npy"):
        glob.unlink()


for path in pathlib.Path("InterviewRecordings").glob("*.transcription"):
    process_transcription(path)
