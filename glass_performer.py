from scamp import *
from live_speech import SpeechBufLoader
import random
import logging

logging.getLogger().setLevel(logging.DEBUG)

speech_loader = SpeechBufLoader()
speech_loader.start()

s = Session()

shard_inst = s.new_osc_part("shardtalk", 57120)

while True:
    shard_inst.play_note(
        [random.randint(40, 80), random.randint(40, 80)],
        [0, random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), 0],
        random.uniform(2, 10),
        {
            "param_pan": [random.random(), random.random()],
            "param_whichbuf": speech_loader.latest_buf()
        },
        blocking=False
    )
    wait(random.uniform(4, 8))
