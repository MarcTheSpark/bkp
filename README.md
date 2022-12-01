# Bauernkriegspanoramamusik

To process the audio simply run `process_audio.py`. This splits the audio into chunks, transcribes those
chunks using Google speech-to-text, and calculates embeddings for those chunks. For each wave file,
it saves a `.textchunks` file (containing the transcribed text), a `.samplocations.npy` file (containing 
the sample locations of each chunk), and a `.vectors.npy` file (containing the embeddings).

You can use `search_recordings.py` to find the chunks that best match a given input text. This is 
done by embedding the input and finding the shortest cosine distance to all of the stored embeddings.

The `SpeechBufLoader` class in `live_speech.py` listens to a live audio input, transcribes it, searches 
for matches, and then sends osc messages to supercollider to loads those matches into buffers.

Finally, `glass_performer.py` is the actual composition itself, utilizing the `SpeechBufLoader` to
listen and load buffers, and playing shards through osc instruments using those buffers.