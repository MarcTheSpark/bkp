import pythonosc.udp_client
from scamp import *
osc_client = pythonosc.udp_client.SimpleUDPClient("127.0.0.1", 57120)

s = Session().run_as_server()


shard_inst = s.new_osc_part("shardtalk", 57120)



for x in range(30):
    osc_client.send_message(r'/loadbuf', [0, "/home/marc/Nextcloud/Concerts/PanoramaMuseum/python/InterviewRecordings/Vatterode_Kunstscheune_KaschubaSimple.wav", 8000000, 400000])
    wait(0.1)
    shard_inst.play_note(60, 1, 4,  blocking=False)
    wait(3)