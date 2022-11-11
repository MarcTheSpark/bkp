import speech_recognition as sr
from read_time_stamp import seek_audio_position
import pythonosc.udp_client
from scamp import *

s = Session().run_as_server()


shard_inst = s.new_osc_part("shardtalk", 57120)

osc_client = pythonosc.udp_client.SimpleUDPClient("127.0.0.1", 57120)


r = sr.Recognizer()
m = sr.Microphone()


with m as source:
    r.adjust_for_ambient_noise(source)
while True:
    with m as source: audio = r.listen(source, phrase_time_limit=5)
    try:
        # recognize speech using Google Speech Recognition
        value = r.recognize_google(audio, language="de")
        found_audio = seek_audio_position(value)

        if found_audio is not None:
            word, audio_pos = found_audio
            print(f"Recognized {word} at audio position {audio_pos}")
            osc_client.send_message(r'/loadbuf', int(audio_pos))
            wait(0.1)
            shard_inst.play_note(60, 1, 4)
        else:
            print("No words recognized in data set")
    except sr.UnknownValueError:
        print("Oops! Didn't catch that")
    except sr.RequestError as e:
        print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
