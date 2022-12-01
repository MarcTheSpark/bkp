import io
import logging
import librosa
import speech_recognition as sr
import pathlib
import yaml
import numpy as np
from itertools import zip_longest
from huggingface_hub.inference_api import InferenceApi


with open("config.yaml", "r") as f:
    conf = yaml.safe_load(f)


def transcribe_audio(input_file):
    input_path = pathlib.Path(input_file)
    sr_audio_file = sr.AudioFile(str(input_path))

    output_transcriptions = input_path.with_suffix(".textchunks")
    output_samps = input_path.with_suffix(".samplocations.npy")

    if output_transcriptions.exists() and output_samps.exists():
        logging.info(f"Recording {input_path.name} already transcribed. Skipping.")
        return

    r = sr.Recognizer()

    chunks, sample_rate = _break_into_chunks(input_file)

    # ---------------------------------------- Transcribe chunks ---------------------------------------

    with open(output_transcriptions, "r+") if output_transcriptions.exists() else open(output_transcriptions, "w") \
            as output_transcriptions_file:
        try:
            transcriptions = output_transcriptions_file.read().strip().split("\n")
        except io.UnsupportedOperation:
            transcriptions = []

        for i, (chunk, transcription) in enumerate(zip_longest(chunks, transcriptions)):
            if transcription is not None:
                # already transcribed this chunk; was saved in the file
                continue
            chunk_start_samp, chunk_end_samp = chunk
            logging.info(f"Transcribing chunk {i+1}/{len(chunks)}")

            with sr_audio_file as source:
                audio = r.record(source, (chunk_end_samp - chunk_start_samp) / sample_rate,
                                 offset=chunk_start_samp / sample_rate)

            recognized_audio = r.recognize_google(audio, language="de")
            output_transcriptions_file.write(recognized_audio + "\n")

    logging.info("Saving transcription locations...")

    np.save(output_samps, np.array(chunks))

    logging.info("Done.")


def _break_into_chunks(input_file):
    input_path = pathlib.Path(input_file)

    logging.info(f"Loading audio file {input_file}...")
    x, sample_rate = librosa.load(str(input_path), sr=None)
    logging.info("done.")

    logging.info("Splitting audio into breakpoints...")
    non_silent_bits = librosa.effects.split(x, top_db=conf["transcription"]["silence_dropoff_threshold"])
    break_points = [segment[0] for segment in non_silent_bits] + [non_silent_bits[-1][1]]

    logging.info(f"Done splitting audio into {len(break_points)} break_points.")

    desired_chunk_length = conf["transcription"]["chunk_length"] * sample_rate
    desired_skip_length = conf["transcription"]["chunk_skip"] * sample_rate

    # ------- Split into chunks of roughly the correct chunk length, offset by roughly the correct skip length --------
    logging.info(f"Creating chunks from breakpoints...")

    bp_start = 0
    chunks = []

    while True:
        chunk_start_samp = break_points[bp_start]
        chunk_end_samp = None
        next_bp_start = None
        for bp_index, bp_samp in enumerate(break_points[bp_start + 1:], start=bp_start + 1):
            if next_bp_start is None and bp_samp - chunk_start_samp >= desired_skip_length:
                next_bp_start = bp_index
            if chunk_end_samp is None and bp_samp - chunk_start_samp >= desired_chunk_length:
                chunk_end_samp = bp_samp
            if next_bp_start is not None and chunk_end_samp is not None:
                chunks.append((chunk_start_samp, chunk_end_samp))
                bp_start = next_bp_start
                break
        else:
            # got to the end of the whole file before finding the end of the chunk and the next start breakpoint
            if chunk_end_samp is not None:
                chunks.append((chunk_start_samp, chunk_end_samp))
            elif break_points[-1] - chunk_start_samp >= desired_chunk_length * 0.7:
                # allow a chunk at the very end that's only 70% as long as it should be
                chunks.append((chunk_start_samp, break_points[-1]))
            break

    logging.info(f"Done creating {len(chunks)} chunks from breakpoints.")
    return chunks, sample_rate


def process_vectors(input_file, batch_size=100):
    input_path = pathlib.Path(input_file)
    vectors_output_path = input_path.parent.joinpath(input_path.stem + ".vectors.npy")
    if vectors_output_path.exists():
        logging.info(f"Vectors for {input_file} already processed. Skipping.")
        return

    with open(input_path.with_suffix(".textchunks")) as f:
        text_chunks = f.read().strip().split("\n")

    text_chunk_beginnings = [" ".join(tc.split(" ")[:conf["embedding"]["num_words_to_front_weight"]])
                             for tc in text_chunks]

    inference = InferenceApi(
        "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
        token="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",
        task="feature-extraction"
    )

    for i in range(0, len(text_chunks), batch_size):
        partial_output_path = input_path.parent.joinpath(input_path.stem + f"-partial-{i:06d}.npy")
        if partial_output_path.exists():
            continue
        logging.info(f"Processing chunks {i}-{i + batch_size - 1} of file {input_path.name}...")
        this_batch_encodings = np.array(inference(inputs=text_chunks[i: i + batch_size]))
        this_batch_beginnings_encodings = np.array(inference(inputs=text_chunk_beginnings[i: i + batch_size]))
        this_batch_encodings = this_batch_beginnings_encodings * conf["embedding"]["front_to_whole_weighting"] +\
                               this_batch_encodings * (1 - conf["embedding"]["front_to_whole_weighting"])

        np.save(partial_output_path, this_batch_encodings)

    logging.info("Concatenating processed chunks and saving all vectors")
    combined_numpy = np.concatenate([
        np.array(np.load(str(glob)))
        for glob in sorted(input_path.parent.glob(f"{input_path.stem}-partial-*.npy"))
    ])

    np.save(vectors_output_path, combined_numpy)

    for glob in input_path.parent.glob(f"{input_path.stem}-partial-*.npy"):
        glob.unlink()
    logging.info("Done.")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    for path in conf["paths"]:
        for file in pathlib.Path(path).glob("*.wav"):
            transcribe_audio(file)
            process_vectors(file, batch_size=conf["embedding"]["processing_batch_size"])
