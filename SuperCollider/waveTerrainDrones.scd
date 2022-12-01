// --------------------------------------------------- WAVE TERRAIN DRONES --------------------------------------------------

(
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


// Set up SynthDef, and fork a few of them!
SynthDef(\wanderWave, {
	arg freq=80, wanderspeed=0.1, amp=0.4, ampModFreq=0.02;
	var ringWidth, circleX, circleY, stereoTexture;
	ringWidth = 0.1;//LFNoise2.ar(0.1);
	circleX = SinOsc.ar(freq, mul:ringWidth);
	circleY = SinOsc.ar(freq, phase:pi/2, mul:ringWidth);
	stereoTexture = [
		WaveTerrain.ar(~waveTerrainBuffer, LFNoise2.ar(wanderspeed) + circleX, LFNoise2.ar(wanderspeed) + circleY, mul: 0.1),
		WaveTerrain.ar(~waveTerrainBuffer, LFNoise2.ar(wanderspeed) + circleX, LFNoise2.ar(wanderspeed) + circleY, mul: 0.1)
	];
	Out.ar(0, LPF.ar(stereoTexture, 4000, mul:amp*SinOsc.kr(ampModFreq)));
}).add;

~drones = Group.new;

~startDrones = {
	s.sync;
	Synth(\wanderWave, [\freq, 55.midicps, \ampModFreq, 0.007], target:~drones);
	Synth(\wanderWave, [\freq, 62.midicps, \ampModFreq, 0.011], target:~drones);
	Synth(\wanderWave, [\freq, 64.midicps, \ampModFreq, 0.013], target:~drones);
};

~startDrones.fork;
)


~startDrones.fork;
~drones.freeAll;








// Wave terrain explorer
(
{
	var freq = 80, ringWidth=0.1;
	var circleX, circleY;

	circleX = SinOsc.ar(freq, mul:ringWidth);
	circleY = SinOsc.ar(freq, phase:pi/2, mul:ringWidth);
	WaveTerrain.ar(b, K2A.ar(MouseX.kr(0, 1)) + circleX, K2A.ar(MouseY.kr(0, 1)) + circleY, mul: 0.1) ! 2
}.play
)