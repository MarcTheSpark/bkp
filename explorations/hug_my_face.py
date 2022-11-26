import logging
from huggingface_hub.inference_api import InferenceApi
import numpy as np
from pathlib import Path


inference = InferenceApi(
    "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
    token = "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",
    task = "feature-extraction"
    )

logging.debug("Trying first inference.")
inference(inputs=["hello"])
logging.debug("Done.")

while True:
    input_sentence = input("Enter a sentence: ")

    print("Connecting to hugging face...", end="")
    input_vec = inference(
        inputs=[input_sentence]
    )[0]
    print("done.")

    all_vecs = np.load(Path.cwd().joinpath("InterviewRecordings/Vatterode_Kunstscheune_KaschubaPart2.vectors.npy"))
    locations = np.load(Path.cwd().joinpath("InterviewRecordings/Vatterode_Kunstscheune_KaschubaPart2.locations.npy"))

    def get_similarities(test_vec, vecs):
        return np.matmul(vecs, test_vec) / np.linalg.norm(vecs, axis=1) / np.linalg.norm(test_vec)


    similarities = get_similarities(input_vec, all_vecs)
    top_indices = np.argsort(similarities)[-5:]
    print(f"Top indices: {top_indices}")
    top_char_locations = locations[top_indices]
    print(similarities[top_indices])

    with open("InterviewRecordings/Vatterode_Kunstscheune_KaschubaPart2.transcription") as f:
        transcription = f.read().replace("\n", " ")

        for start, finish in top_char_locations:
            print(transcription[start: finish])
