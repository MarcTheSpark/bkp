(

s.options.memSize = 2.pow(15);
~fftsize = 4096;

s.waitForBoot {

    {
        var irbuffers, bufsizes, setupFunction, effects;

		MarcUtilities.setOutputLimiter(0.5, 0.5);

		irbuffers = Array.fill(2, { |i|
			Buffer.readChannel(s, "ConcertHall.wav".resolveRelative, channels:[i]);
		});
        s.sync;

		bufsizes = Array.fill(2, { |i|
			PartConv.calcBufSize(~fftsize, irbuffers[i]);
		});

		// ~numpartitions= PartConv.calcNumPartitions(~fftsize, irbuffers[0]);

        ~irspectrums = Array.fill(2, { |i|
			var spec = Buffer.alloc(s, bufsizes[i], 1);
			spec.preparePartConv(irbuffers[i], ~fftsize);
			spec;
		});

		~reverbBus = Bus.new(server: s);

		s.sync;

		SynthDef(\verb, { |out=0|
			var input = In.ar(~reverbBus, 2);
			Out.ar(out, PartConv.ar(input, ~fftsize, ~irspectrums, 0.002));
		}).add;

		SynthDef(\test, {
			var input = Pan2.ar(SinOsc.ar(SinOsc.kr(0.3, mul:300, add:600), mul:0.1), SinOsc.kr(0.2));
			Out.ar(~reverbBus, input);
		}).add;

        s.sync;

		2.do({ |i|
			irbuffers[i].free; // don't need time domain data anymore, just needed spectral version
		});

		setupFunction = {
			effects.free;
			effects = Group.new(s.defaultGroup, addAction:'addAfter');
			Synth(\verb, target:effects);
		};
		ServerTree.add(setupFunction);
		setupFunction.value;
    }.fork;

}
)

~target = Buffer.read(s, Platform.resourceDir +/+ "sounds/a11wlk01.wav");

Synth(\test);