b = Buffer.cueSoundFile(s, "/home/marc/Recent & relevant/PanoramaMuseum/python/Vatterode_Kunstscheune_KaschubaSimple.wav", 0, 1);
b = Buffer.read(s, "/home/marc/Recent & relevant/PanoramaMuseum/python/Vatterode_Kunstscheune_KaschubaSimple.wav", 14893568-30000, 200000)
b.free
{PlayBuf.ar(1, b) ! 2}.play
Buffer.read(s, "/home/marc/Recent & relevant/PanoramaMuseum/python/Vatterode_Kunstscheune_KaschubaSimple.wav",100000, bufnum:b)
s.sendMsg([\b_read, b.bufnum, "/home/marc/Recent & relevant/PanoramaMuseum/python/Vatterode_Kunstscheune_KaschubaSimple.wav", 0, 1])

