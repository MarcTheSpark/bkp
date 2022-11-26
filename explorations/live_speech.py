import threading
import time
from pathlib import Path
from threading import Thread
import speech_recognition as sr
from read_time_stamp import seek_phrase
import pythonosc.udp_client
import logging

logging.getLogger().setLevel(logging.DEBUG)

print(sr.Microphone.list_microphone_names())

class SpeechBufLoader(Thread):

    def __init__(self, osc_send_ip="127.0.0.1", osc_send_port=57120, phrase_time_limit=3,
                 buf_length=48000*10, num_bufs=10, device_name=None):
        self.r = sr.Recognizer()
        self.m = sr.Microphone(
            sr.Microphone.list_microphone_names().index(device_name)
            if device_name is not None else None
        )
        self.osc_client = pythonosc.udp_client.SimpleUDPClient(osc_send_ip, osc_send_port)
        self.phrase_time_limit = phrase_time_limit
        self.buf_length = buf_length
        self.num_bufs = num_bufs
        self._current_buf = 0
        self._buf_lock = threading.Lock()
        super().__init__()

    def run(self) -> None:
        # while True:
        #     value = input("Type something:")
        #     found_audio = seek_phrase(value)
        #
        #     if found_audio is not None:
        #         word, file_path, audio_pos = found_audio
        #         logging.debug(f"Recognized {word} at audio position {audio_pos} in {Path(file_path).name}")
        #         self.osc_client.send_message(r'/loadbuf',
        #                                      [self._next_buf, file_path, int(audio_pos), self.buf_length])
        #         threading.Thread(target=self._delayed_increment_buf).start()
        #     else:
        #         logging.debug("No words recognized in data set")
        with self.m as source:
            self.r.adjust_for_ambient_noise(source)
        while True:
            with self.m as source:
                audio = self.r.listen(source, phrase_time_limit=3)
            try:
                # recognize speech using Google Speech Recognition
                value = self.r.recognize_google(audio, language="de")
                logging.debug(f"Recognized '{value}'")

                found_audio = seek_phrase(value)

                if found_audio is not None:
                    word, file_path, audio_pos = found_audio
                    logging.debug(f"Found {word} at audio position {audio_pos} in {Path(file_path).name}")
                    self.osc_client.send_message(r'/loadbuf',
                                                 [self._current_buf, file_path, int(audio_pos), self.buf_length])
                    threading.Thread(target=self._delayed_increment_buf).start()
                else:
                    logging.debug("No words recognized in data set")
            except sr.UnknownValueError:
                logging.debug("Oops! Didn't catch that")
            except sr.RequestError as e:
                logging.debug("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))

    def _delayed_increment_buf(self):
        time.sleep(2)
        with self._buf_lock:
            self._current_buf = (self._current_buf + 1) % self.num_bufs

    def latest_buf(self):
        return self.get_buf(-1)

    def get_buf(self, offset=0):
        with self._buf_lock:
            return (self._current_buf + offset) % 10


if __name__ == '__main__':
    sbl = SpeechBufLoader()
    sbl.start()
    while True:
        # print(sbl.latest_buf())
        time.sleep(0.1)
