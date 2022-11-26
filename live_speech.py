import random
import threading
import time
from threading import Thread
import speech_recognition as sr
from search_recordings import find_best_matches_in_all_recordings
from pythonosc import udp_client
from pythonosc import osc_server
from pythonosc.dispatcher import Dispatcher
import logging

# logging.getLogger().setLevel(logging.DEBUG)
# print(sr.Microphone.list_microphone_names())


class SpeechBufLoader(Thread):

    def __init__(self, osc_send_ip="127.0.0.1", osc_send_port=57120, osc_receive_ip="127.0.0.1", osc_receive_port=60606,
                 phrase_time_limit=None, buf_length=48000*10, num_bufs=10, device_name=None):
        self.r = sr.Recognizer()
        self.m = sr.Microphone(
            sr.Microphone.list_microphone_names().index(device_name)
            if device_name is not None else None
        )
        self.osc_client = udp_client.SimpleUDPClient(osc_send_ip, osc_send_port)
        self.set_up_buf_loaded_listener(osc_receive_ip, osc_receive_port)
        self.phrase_time_limit = phrase_time_limit
        self.buf_length = buf_length
        self.num_bufs = num_bufs
        self._next_buf_to_fill = 0
        self._bufs_lock = threading.Lock()
        self._fresh_bufs = []
        self.buf_loaded_callback = lambda *args: None
        super().__init__(daemon=True)

    def set_up_buf_loaded_listener(self, ip, port):
        dispatcher = Dispatcher()
        dispatcher.map("/buf_loaded", self.register_loaded_buf)
        server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
        threading.Thread(target=server.serve_forever, daemon=True).start()

    def run(self) -> None:
        logging.debug(f"Adjusting for ambient noise")
        with self.m as source:
            self.r.adjust_for_ambient_noise(source)
        while True:
            logging.debug(f"Listening to audio")
            with self.m as source:
                audio = self.r.listen(source, phrase_time_limit=self.phrase_time_limit)
            logging.debug(f"Sending audio to Google for recognition")
            try:
                # recognize speech using Google Speech Recognition
                value = self.r.recognize_google(audio, language="de")
                logging.debug(f"Recognized '{value}'")

                matches = find_best_matches_in_all_recordings(value)

                logging.debug(f"Top matches: {[match.text for match in matches]}")
                for match in reversed(matches):
                    self.osc_client.send_message(
                        r'/loadbuf',
                        [self._next_buf_to_fill, str(match.file_path), int(match.recording_start_sample) - 48000,
                         10 * 48000]
                    )
                    self._next_buf_to_fill = (self._next_buf_to_fill + 1) % self.num_bufs

            except sr.UnknownValueError:
                logging.debug("Oops! Didn't catch that")
            except sr.RequestError as e:
                logging.debug("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))

    def register_loaded_buf(self, osc_address, newly_loaded_buf):
        with self._bufs_lock:
            while newly_loaded_buf in self._fresh_bufs:
                self._fresh_bufs.remove(newly_loaded_buf)
            self._fresh_bufs.append(newly_loaded_buf)

    def pop_fresh_buf(self):
        with self._bufs_lock:
            if len(self._fresh_bufs) > 0:
                return self._fresh_bufs.pop(-1)
        return None

    def get_fresh_bufs(self):
        return tuple(self._fresh_bufs)


if __name__ == '__main__':
    sbl = SpeechBufLoader()
    sbl.start()
    while True:
        print(sbl.get_fresh_bufs())
        if random.random() < 0.1:
            print(f"popping {sbl.pop_fresh_buf()}")
        time.sleep(0.5)
