from scamp import *
from live_speech import SpeechBufLoader
import random
import logging

logging.getLogger().setLevel(logging.DEBUG)

speech_loader = SpeechBufLoader()
speech_loader.start()

s = Session()

shard_talk = s.new_osc_part("shardtalk", 57120)
pure_shards = s.new_osc_part("pureshard", 57120)

while True:
    new_buf = speech_loader.pop_fresh_buf()
    if new_buf is None:
        pure_shards.play_note(
            [random.randint(40, 80), random.randint(40, 80)],
            [0, random.uniform(0.1, 0.7), random.uniform(0.1, 0.7), 0],
            random.uniform(2, 10),
            {
                "param_pan": [random.uniform(-1, 1), random.uniform(-1, 1)],
            },
            blocking=False
        )
        wait(random.uniform(4, 8))
    else:
        shard_talk.play_note(
            [random.randint(50, 70), random.randint(50, 70)],
            [random.uniform(0.8, 1.0), random.uniform(1.0, 1.0), 0],
            random.uniform(6, 13),
            {
                "param_pan": [random.uniform(-1, 1), random.uniform(-1, 1)],
                "param_whichbuf": new_buf
            },
            blocking=False
        )
        wait(random.uniform(4, 8))
