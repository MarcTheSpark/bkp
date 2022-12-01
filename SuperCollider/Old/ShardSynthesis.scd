(
a = PathName("Shards".resolveRelative);
~shardBuffers = List.new;
~flatEnv = Buffer.loadCollection(s, [1, 1]);
~speechBuf = Buffer.read(s, "165186__blaukreuz__global-village-odenwald-bergstrasse.wav".resolveRelative);

a.filesDo({
	arg file;
	~shardBuffers.add(Buffer.alloc(s, s.sampleRate * 1, 1,).readChannel(file.asAbsolutePath, channels: [0]));
});
)


(
{
	var trig, volume, shardGrains, speech;
	trig = Dust.kr(MouseX.kr(10, 100));
	volume = K2A.ar(MouseX.kr);
	shardGrains = GrainBuf.ar(
		2,
		trig,
		dur:1,
		sndbuf: Select.kr(TIRand.kr(0, ~shardBuffers.size, trig),~shardBuffers),
		envbufnum: ~flatEnv,
		rate: Demand.kr(trig, 0, Dwhite(0.5, 2)),
	) * volume;
	speech = PlayBuf.ar(2, ~speechBuf, loop:1);
	Convolution.ar(speech, shardGrains, framesize:2048) + shardGrains * (0.1 + Amplitude.ar(speech));
}.play
)


(
ScampUtils.instrumentFromSynthDef(
	SynthDef(\shardtalk, {
		arg gate=1, volume=1, pitch=60, whichbuf=0;
		var trig, shardGrains, speech, output, pitchRateMul;
		trig = Dust.kr(volume.linlin(0, 1, 20, 100));
		pitchRateMul = pitch.linexp(0, 127, 0.5, 2);
		shardGrains = GrainBuf.ar(
			2,
			trig,
			dur:1,
			sndbuf: Select.kr(TIRand.kr(0, ~shardBuffers.size, trig),~shardBuffers),
			envbufnum: ~flatEnv,
			rate: Demand.kr(trig, 0, Dwhite(0.8, 1.2) * pitchRateMul),
		) * volume;
		speech = PlayBuf.ar(2, Select.kr(whichbuf, ~speechBufs), rate: pitchRateMul, loop:1) ! 2;
		output = Convolution.ar(speech, shardGrains, framesize:2048) + shardGrains * (0.1 + Amplitude.ar(speech));
		Out.ar(
			0,
			output * EnvGate.new(gate:gate, fadeTime:0.1)
		);
	})
)
)


(
n = NetAddr("127.0.0.1", NetAddr.langPort); // local machine
~firstBuf = s.nextBufferNumber(10)
OSCFunc.new({
	|msg, time, addr, recvPort|
	var whichbuf = msg[1], path = msg[2], samp = msg[3], length = msg[4];
	// load this into a new buffer and put it at the front of the cue
	Buffer.read(s, path, samp - 60000, length, bufnum:whichbuf);
}, '/loadbuf');
)

/*
(
~speechBufs = List.new;
n = NetAddr("127.0.0.1", NetAddr.langPort); // local machine
~firsts.nextBufferNumber(10)
OSCFunc.new({
	|msg, time, addr, recvPort|
	var whichbuf = msg[1], path = msg[2], samp = msg[3], length = msg[4];
	// load this into a new buffer and put it at the front of the cue
	~speechBufs.addFirst(Buffer.read(s, path, samp - 60000, length));
	if(~speechBufs.size > 3, {
		// if the cue is getting long, pop its last one, and free it after a bit (in case any synths are still using it)
		var oldest = ~speechBufs.pop;
		{8.0.yield; oldest.free}.fork;
	});
}, '/loadbuf');
)
*/