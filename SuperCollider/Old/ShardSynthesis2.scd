(
~flatEnv = Buffer.loadCollection(s, [1, 1]);
~decayEnv = Buffer.loadCollection(s, Env.perc.asSignal);
~shardBuffers = List.new;
"Loading shards...".post;
PathName("Shards".resolveRelative).filesDo({
	arg file;
	~shardBuffers.add(Buffer.alloc(s, s.sampleRate * 1, 1,).readChannel(file.asAbsolutePath, channels: [0]));
});
"done.".postln;
)

(
~firstSpeechBuf = s.nextBufferNumber(10);
"Reserved buffers %-%".format(~firstSpeechBuf, ~firstSpeechBuf + 9).postln;
~oscToPython = NetAddr.new("127.0.0.1", 60606);
OSCFunc.new({
	|msg, time, addr, recvPort|
	var whichbuf = msg[1], path = msg[2], samp = msg[3], length = msg[4];
	{
		// load this into a new buffer and put it at the front of the cue
		"Loading snipped from %".format([path]).postln;
		Buffer.read(s, path, samp - 65000, length, bufnum:(~firstSpeechBuf + whichbuf));
		s.sync;
		~oscToPython.sendMsg(\buf_loaded, whichbuf);
	}.fork
}, '/loadbuf');
"Listening for /loadbuf messages".postln;
)



(
ScampUtils.instrumentFromSynthDef(
	SynthDef(\shardtalk, {
		arg gate=1, volume=1, pitch=60, whichbuf=0, pan=0, panWidth=0.1;
		var trig, shardGrains, speech, output, pitchRateMul;
		trig = Dust.kr(volume.linlin(0, 1, 20, 100));
		pitchRateMul = pitch.linexp(0, 127, 0.5, 2);
		shardGrains = GrainBuf.ar(
			2,
			trig,
			dur:1,
			sndbuf: Select.kr(TIRand.kr(0, ~shardBuffers.size, trig),~shardBuffers),
			envbufnum: ~flatEnv,
			pan: Demand.kr(trig, 0, Dwhite(pan - panWidth, pan + panWidth)),
			rate: Demand.kr(trig, 0, Dwhite(0.8, 1.2) * pitchRateMul),
		) * volume;
		speech = PlayBuf.ar(1, ~firstSpeechBuf + whichbuf, rate: pitchRateMul, loop:1) ! 2;
		output = (Convolution.ar(speech, shardGrains, framesize:2048) + shardGrains) * (0.1 + Amplitude.ar(speech));
		output = output * EnvGate.new(gate:gate, fadeTime:0.1);

		Out.ar(
			0,
			output
		);
	}), target: ~sources
);
ScampUtils.instrumentFromSynthDef(
	SynthDef(\pureshard, {
		arg gate=1, volume=1, pitch=60, pan=0, panWidth=0.1;
		var trig, shardGrains, speech, output, pitchRateMul;
		trig = Dust.kr(volume.linlin(0, 1, 20, 100));
		pitchRateMul = pitch.linexp(0, 127, 0.5, 2);
		shardGrains = GrainBuf.ar(
			2,
			trig,
			dur:1,
			sndbuf: Select.kr(TIRand.kr(0, ~shardBuffers.size, trig),~shardBuffers),
			envbufnum: ~flatEnv,
			pan: Demand.kr(trig, 0, Dwhite(pan - panWidth, pan + panWidth)),
			rate: Demand.kr(trig, 0, Dwhite(0.8, 1.2) * pitchRateMul),
		) * volume;

		output = shardGrains * EnvGate.new(gate:gate, fadeTime:0.1);

		Out.ar(
			0,
			output
		);
	}), target: ~sources
)
)

MarcUtilities.doSpeakerTest(outputLevel:0.05)

// Add G, D E Drones fading in and out?
// speech creates shattering
// sometimes speech gets overwhelming, and then maybe the closest match comes out solo
// mix in original audio
// response to microphone input

//
//
//
// ~distanceSimulator = {
// 	|in, distance, directOut, reverbOut|
// 	var lowPassed = LPF.ar(in, distance.linexp(0, 1, 2000, 40));
// 	var mixedWithLowPass = (lowPassed * distance) + (in * (1 - distance));
// 	Out.ar(directOut, mixedWithLowPass * (1 - distance));
// 	Out.ar(reverbOut, mixedWithLowPass * distance);
// }
//
// (
// ~setup = {
// 	{
// 		s.sync;
// 		~verbBus = Bus.audio(s, 2);
// 		~effects = Group.new(addAction:'addToTail');
// 		s.sync;
// 		{
// 			GVerb.ar(In.ar(~verbBus),100, 5, drylevel:0);
// 		}.play(target:~effects)
// 	}.fork;
// };
// ~setup.value;
// CmdPeriod.add(~setup);
// )
// CmdPeriod.removeAll
//
// {EnvGen.ar(Env.perc, Impulse.kr(0.8)) * WhiteNoise.ar(0.1)}.play
// CommandPeriod


// (
// {
// 	var trig = Dust.kr(80);
// 	GrainBuf.ar(
// 		2,
// 		trig,
// 		pos: Sweep.ar(rate:s.sampleRate/BufFrames.ir(~firstSpeechBuf)),
// 		dur: Demand.kr(trig, 0, Dwhite(0.03, 0.08)),
// 		pan: Demand.kr(trig, 0, Dwhite(-1, 1)),
// 		sndbuf: ~firstSpeechBuf,
// 		envbufnum: ~decayEnv,
// 	)
// }.play
// )