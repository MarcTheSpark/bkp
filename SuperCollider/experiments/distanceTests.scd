(
~distance = {
	|inUGen, distance, directOut=0, reverbOut=0|
	var lpffedUp = LPF.ar(inUGen, distance.linexp(0, 1, 15000, 50));
	var lowPassProportion = distance.linlin(0, 0.5, 0, 1);
	var proportionallyLowPassed = (inUGen * (1 - lowPassProportion)) + (lpffedUp * lowPassProportion);
	var directProportion = distance.linlin(0, 1, 1, 0.2);
	var verbedProportion = distance.linlin(0, 1, 0.2, 1);
	Out.ar(directOut, proportionallyLowPassed * directProportion);
	Out.ar(reverbOut, proportionallyLowPassed * verbedProportion);
};

~verbbus = Bus.audio(s, 2);
{Greyhole.ar(In.ar(~verbbus), 0.3, damp: 0.3, size:0.9, feedback:0.7)}.play;
)

(
{
	var crackle = GrainIn.ar(
		2,
		Dust.kr(80),
		0.004,
		WhiteNoise.ar,
		pan:WhiteNoise.kr(0.3, MouseX.kr(-1, 1)),
		mul:0.2
	);
	~distance.value(crackle, MouseY.kr, 0, ~verbbus);
}.play
)