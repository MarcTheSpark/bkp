MarcUtilities.setOutputLimiter
MarcUtilities.doSpeakerTest(outputLevel:0.05)


~startDrones.value;
~shardClouds.freeAll;
~drones.freeAll;
~reset.value;


(
// GLOBAL VARS
~numSpeechBufs = 10;
~reverbFFTsize = 4096;
~reverbIR = "SoundFiles/IRs/ConcertHall.wav";
s.options.memSize = 2.pow(15);


// SETUP FUNCTIONS
~loadEnvelopesAndShards = {
	~flatEnv = Buffer.loadCollection(s, [1, 1]);
	~decayEnv = Buffer.loadCollection(s, Env.perc.asSignal);
	~shardBuffers = List.new;
	PathName("SoundFiles/Shards".resolveRelative).filesDo({
		arg file;
		~shardBuffers.add(Buffer.alloc(s, s.sampleRate * 1, 1,).readChannel(file.asAbsolutePath, channels: [0]));
	});
};

~setUpGroups = {
	~sources = Group.new;
	~drones = Group.new(target:~sources);
	~shardClouds = Group.new(target: ~sources);
	~reverb = Group.after(~sources);
	~reverbBus = Bus.audio(s, 2);
};

~setUpSpeechBufs = {
	~firstSpeechBuf = s.nextBufferNumber(~numSpeechBufs);
	// "Reserved buffers %-%".format(~firstSpeechBuf, ~firstSpeechBuf + ~numSpeechBufs - 1).postln;
	~oscToPython = NetAddr.new("127.0.0.1", 60606);
	OSCFunc.new({
		|msg, time, addr, recvPort|
		var whichbuf = msg[1], path = msg[2], samp = msg[3], length = msg[4];
		{
			// load this into a new buffer and put it at the front of the cue
			"Loading snipped from %".format([path]).postln;
			Buffer.read(s, path, samp, length, bufnum:(~firstSpeechBuf + whichbuf));
			s.sync;
			~oscToPython.sendMsg(\buf_loaded, whichbuf);
		}.fork
	}, '/loadbuf');
	// "Listening for /loadbuf messages".postln;
};

~loadWaveTerrainArray = {
	// Set up wave terrain array and load into buffer
	var width= 100; //= num cols
	var height=100; //=num rows, though indexing bottom to top; i.e., standard Cartesian co-ordinates
	var waveTerrainArray = Array.fill(width*height,{arg i;
		var xnow, ynow, x, y, r;

		xnow= i%width;
		ynow= (i-xnow).div(width);

		x=xnow/width * 2 - 1;
		y=ynow/height * 2 - 1;

		r = x.pow(2) + y.pow(2);

		((r < 1) && (r > 0)).if(r * (pi/2/r).cos, 0);
	});

	~waveTerrainBuffer = Buffer.sendCollection(s, waveTerrainArray, 1);
};

~loadAndPrepImpulseResponses = {
	{
		var irbuffers, bufsizes;
		irbuffers = Array.fill(2, { |i|
			Buffer.readChannel(s, ~reverbIR.resolveRelative, channels:[i]);
		});
		s.sync;

		bufsizes = Array.fill(2, { |i|
			PartConv.calcBufSize(~reverbFFTsize, irbuffers[i]);
		});


		~irspectrums = Array.fill(2, { |i|
			var spec = Buffer.alloc(s, bufsizes[i], 1);
			spec.preparePartConv(irbuffers[i], ~reverbFFTsize);
			spec;
		});
	}.fork;
};

~startDrones = {
	Synth(\wanderWave, [\out, ~reverbBus, \freq, 55.midicps, \ampModFreq, 0.007], target:~drones);
	Synth(\wanderWave, [\out, ~reverbBus, \freq, 62.midicps, \ampModFreq, 0.011], target:~drones);
	Synth(\wanderWave, [\out, ~reverbBus, \freq, 64.midicps, \ampModFreq, 0.013], target:~drones);
};

~setUpSynthDefs = {
	~distance = {
		|inUGen, distance, directOut=0, reverbOut=0|
		var lpffedUp = LPF.ar(inUGen, distance.linexp(0, 1, 15000, 400));
		var lowPassProportion = distance.linlin(0, 0.5, 0, 1);
		var proportionallyLowPassed = (inUGen * (1 - lowPassProportion)) + (lpffedUp * lowPassProportion);
		var directProportion = distance.linlin(0, 1, 1, 0.2);
		var verbedProportion = distance.linlin(0, 1, 0.2, 1);
		Out.ar(directOut, proportionallyLowPassed * directProportion);
		Out.ar(reverbOut, proportionallyLowPassed * verbedProportion);
	};


	ScampUtils.instrumentFromSynthDef(
		SynthDef(\shardtalk, {
			arg out=0, gate=1, volume=1, pitch=60, whichbuf=0, pan=0, panWidth=0.1, distance=0;
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

			~distance.value(output, distance, out, ~reverbBus);
		}), target: ~shardClouds
	);

	ScampUtils.instrumentFromSynthDef(
		SynthDef(\pureshard, {
			arg out=0, gate=1, volume=1, pitch=60, pan=0, panWidth=0.1, distance=0;
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
			~distance.value(output, distance, out, ~reverbBus);
		}), target: ~shardClouds
	);

	SynthDef(\responsiveShards, {
		arg out=0, gate=1, volume=1, pitch=60, pan=0, panWidth=0.5, inputAmplitudeThreshold=0.002;
		var trig, shardGrains, speech, output, pitchRateMul;
		volume = volume * (Amplitude.ar(SoundIn.ar, releaseTime: 2.0) - inputAmplitudeThreshold).max(0).pow(0.5);
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
			out,
			output
		);
	}).add;

	SynthDef(\wanderWave, {
		arg out=0, freq=80, wanderspeed=0.1, amp=0.4, ampModFreq=0.02;
		var ringWidth, circleX, circleY, stereoTexture;
		ringWidth = 0.1;//LFNoise2.ar(0.1);
		circleX = SinOsc.ar(freq, mul:ringWidth);
		circleY = SinOsc.ar(freq, phase:pi/2, mul:ringWidth);
		stereoTexture = [
			WaveTerrain.ar(~waveTerrainBuffer, LFNoise2.ar(wanderspeed) + circleX, LFNoise2.ar(wanderspeed) + circleY, mul: 0.1),
			WaveTerrain.ar(~waveTerrainBuffer, LFNoise2.ar(wanderspeed) + circleX, LFNoise2.ar(wanderspeed) + circleY, mul: 0.1)
		];
		Out.ar(out, LPF.ar(stereoTexture, 4000, mul:amp*SinOsc.kr(ampModFreq)));
	}).add;

	SynthDef(\verb, {
		arg out=0;
		var signalIn = In.ar(~reverbBus);
		Out.ar(out, PartConv.ar(signalIn, ~reverbFFTsize, ~irspectrums, 0.05));
	}).play(target:~reverb);
};

~reset = {
	~shardClouds.freeAll;
	~drones.freeAll;
	~numSpeechBufs.do({ |i|
		Buffer.new(s, numChannels:1, bufnum:~firstSpeechBuf + i).zero;
	});
};


s.reboot;
s.waitForBoot({
	{
		"Loading envelopes and shards...".post;
		~loadEnvelopesAndShards.value; s.sync;
		"done.".postln;
		"Setting up groups...".post;
		~setUpGroups.value;	s.sync;
		"done.".postln;
		"Setting up speech buffers...".post;
		~setUpSpeechBufs.value; s.sync;
		"done.".postln;
		"Creating and loading wave terrain...".post;
		~loadWaveTerrainArray.value; s.sync;
		"done.".postln;
		"Loading and prepping impulse responses...".post;
		~loadAndPrepImpulseResponses.value; s.sync;
		"done.".postln;
		"Setting up SynthDefs...".post;
		~setUpSynthDefs.value; s.sync;
		"done.".postln;
		"Starting drones...".post;
		~startDrones.value; s.sync;
		"done.".postln;
	}.fork;
});
)


Synth(\responsiveShards)

{Out.ar(~reverbBus, SinOsc.ar * EnvGen.ar(Env.perc(releaseTime:0.1)) * 0.6)}.play

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
