use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicBool;
use std::time::{Duration};
use std::{io, fs};
use std::path::{Path, PathBuf};
use std::fs::File;

use fundsp::hacker32::{mul, constant, pulse, declick, envelope, sine, timer, tag};
use fundsp::oscillator;
use fundsp::prelude::{AudioUnit32, lerp};
use midly::MidiMessage;
use serde::{Deserialize, Serialize};

use midi_player::Synth;

#[derive(Clone, Copy)]
struct ASDREnvelope {
    attack_duration: f32,
    decay_duration: f32, 
    release_duration: f32,

    initial_db: f32,
    peak_db: f32,
    sustain_db: f32,
}

impl ASDREnvelope {
    fn new() -> Self { 
        Self {
            attack_duration: 0.1,
            decay_duration: 0.4,
            release_duration: 0.3,

            initial_db: 0.0,
            peak_db: 1.0,
            sustain_db: 0.6
        }
    }

    fn is_envelope_finished(&self, time_since_start: f32, time_released: Option<f32>) -> bool {
        match time_released {
            None => false,
            Some(released) => (time_since_start - released) > self.release_duration
        }
    }

    fn get_amp_at_time(&self, time_since_start: f32, time_released: Option<f32>) -> f32 {
        // Determine what stage of the envelope we are at.
        match time_released {
            None => {
                let time_in_attack_phase = time_since_start;
                let time_in_decay_phase = time_in_attack_phase - self.attack_duration;

                if time_in_attack_phase < self.attack_duration {
                    // Linear interpolation to peak db from initial db
                    lerp(self.initial_db, self.peak_db, time_in_attack_phase / self.attack_duration)
                } else if time_in_decay_phase < self.decay_duration {
                    // Linear interpolation to sustain db from peak db
                    lerp(self.peak_db, self.sustain_db, time_in_decay_phase / self.decay_duration)
                } else {
                    // The sustain volume
                    self.sustain_db
                }
            },
            // Either still in the release phase or finished.
            Some(release_time) => {
                let time_in_release = time_since_start - release_time;
                lerp(self.sustain_db, 0.0, time_in_release / self.release_duration).max(0.0)
            }
        }
    }
}

pub enum VoiceTag {
    Time
}

pub enum OscilatorWaveKind {
    Sine,
    Pulse(f32),
    Square,
    Saw,
    WhiteNoise,
    PinkNoise
}

pub struct OscillatorConfig {
    kind: OscilatorWaveKind,
    volume: f32
}

pub struct VoiceConfig {
    // Oscilator 1 type
    oscillators: [Option<OscillatorConfig>; 3],

    // The envelope to use for this voice.
    volume_envelope: ASDREnvelope
}

impl VoiceConfig {
    pub fn new() -> Self {
        let mut oscillators = [
            Some(OscillatorConfig {
                kind: OscilatorWaveKind::Saw,
                volume: 1.0
            }),
            None,
            None
        ];

        Self {
            oscillators,
            volume_envelope: ASDREnvelope::new()
        }
    }
}

pub struct Voice {
    // The audio node to get our samples from.
    audio_node: Box<dyn AudioUnit32>,

    // So that we can query the envelope settings.
    envelope: ASDREnvelope,
    // So the asdr envelope can know when the note stopped being played.
    time_released: Arc<Mutex<Option<f32>>>
}

impl Voice {
    pub fn from_config(config: &VoiceConfig, velocity: f32) -> Self {
        // An atomic bool that we can use to control the envelope from outside.
        let time_released = Arc::new(Mutex::new(None));

        // Create an envelope node that can get the value of the ASDREnvelope as the note plays.
        let asdr = config.volume_envelope.clone(); // Copy the envelope settings.
        let volume_modulation = {
            let time_released = time_released.clone();
            envelope(move |t| {
                // Return the envelope value.
                let time_released = time_released.lock().unwrap();
                (&asdr).get_amp_at_time(t, *time_released)
            })
        };
        
        // Create the oscillators
        let create_oscillator = |config: OscillatorConfig| -> Option<Box<dyn AudioUnit32>> {
            match config.kind {
                OscilatorWaveKind::Sine => return Some(Box::new(
                    sine()
                )),
                _ => return None
            }
        }

        let audio_node = 
            timer(VoiceTag::Time as i64) |
            (
                //((mul(1.0) | constant(0.5)) >> pulse()) + // Oscillator 1
                (sine()) // Oscillator 2
            )
            >> declick() // Filter to remove clicking from the start of the voice.
            * volume_modulation // Volume modulation by the envelope function.
            * constant(velocity); // Volume adjusted by velocity of note.
        Self {
            audio_node: Box::new(audio_node),
            envelope: asdr,
            time_released,
        }
    }

    pub fn get_tag_value(&self, tag: VoiceTag) -> Option<f64> {
        self.audio_node.get(tag as i64)
    }

    pub fn set_released(&mut self) {
        let mut time_released = self.time_released.lock().unwrap();
        let current_time = self.get_tag_value(VoiceTag::Time).expect("The time parameter should be available!");
        *time_released = Some(current_time as f32);
    }

    pub fn is_finished(&self) -> bool {
        let time_released = self.time_released.lock().unwrap();
        let current_time = self.get_tag_value(VoiceTag::Time).expect("The time parameter should be available!");
        self.envelope.is_envelope_finished(current_time as f32, *time_released)
    }
}

pub struct VoiceBank {
    voice_configs: Vec<VoiceConfig>
}

impl VoiceBank {
    pub fn new() -> Self {
        let mut bank = Self {
            voice_configs: Vec::new()
        };

        let config = VoiceConfig::new();
        bank.voice_configs.push(config);
        bank
    }

    pub fn get_config(&self, index: usize) -> &VoiceConfig {
        &self.voice_configs[0]
    }
}

pub struct Key {
    channel: usize,
    midi_note: usize,
    samples_played: usize,
    samples_stopped_at: Option<usize>,

    // The voice to use to generate samples.
    voice: Voice
}

impl Key {
    pub fn get_samples(&mut self, output_channel_count: usize, output: &mut [f32]) {
        const MAX_CHANNEL_COUNT: usize = 10;

        let freq = midi_player::midi_note_to_freq(self.midi_note as u8);
        let inputs = [freq];

        let voice_output_count = self.voice.audio_node.outputs();
        if voice_output_count > MAX_CHANNEL_COUNT {
            panic!("Voice has too many outputs / channels!");
        }

        // Generate samples for each of these buffers.
        let mut temp_buf = [0.0; MAX_CHANNEL_COUNT];

        let mut sampled = 0;
        for frame in output.chunks_exact_mut(output_channel_count) {
            // The output from the audio node.
            self.voice.audio_node.tick(&inputs, &mut temp_buf[..voice_output_count]);

            for channel in 0..output_channel_count {
                if channel >= voice_output_count {
                    frame[channel] += temp_buf[voice_output_count-1];
                } else {
                    frame[channel] += temp_buf[channel];
                }
            }

            sampled += 1;
        }

        self.samples_played += sampled;
    }
}

pub struct SimpleSynth {
    bank: VoiceBank,
    keys: Vec<Key>,

    // Map of channels to midi voices.
    voices: HashMap<usize, usize>,
    sample_rate: usize
}

impl SimpleSynth {
    pub fn new(bank: VoiceBank, sample_rate: usize) -> Self {
        Self {
            bank,
            keys: Vec::new(),
            voices: HashMap::new(),
            sample_rate
        }
    }

    fn note_on(&mut self, channel: usize, midi_note: usize, vel: usize) {
        tracing::debug!("Key {} on at {} velocity", midi_note, vel);

        // Get the voice config for this channel.
        let config = self.bank.get_config(channel);
        let mut voice = Voice::from_config(config, vel as f32 / 127.0);
        voice.audio_node.reset(Some(self.sample_rate as f64));

        // Add the note.
        self.keys.push(Key {
            channel,
            midi_note, 
            samples_played: 0,
            samples_stopped_at: None,
            voice
        });
    }

    fn note_off(&mut self, channel: usize, midi_note: usize) {
        tracing::debug!("Key {} off", midi_note);

        // Find the note and record the time it stopped.
        for key in self.keys.iter_mut() {
            if key.channel == channel && 
               key.midi_note == midi_note && 
               key.samples_stopped_at.is_none() {

                key.samples_stopped_at = Some(key.samples_played);

                // Tell the voice to stop.
                key.voice.set_released();
            }
        }
    }

    fn purge_finished_notes(&mut self) {
        // Get rid of any notes that have played more than the threshold of samples past where they stopped.
        self.keys.retain(|key| {
            !key.voice.is_finished()
        });
    }
}

impl Synth for SimpleSynth {
    fn midi_message(&mut self, channel: usize, message: MidiMessage) {
        match message {
            MidiMessage::NoteOn { key, vel } => {
                if vel == 0 {
                    self.note_off(channel, key.as_int() as usize);
                } else {
                    self.note_on(channel, key.as_int() as usize, vel.as_int() as usize);
                }
            },
            MidiMessage::NoteOff { key, vel: _ } => {
                self.note_off(channel, key.as_int() as usize);
            },
            MidiMessage::ProgramChange { program } => {
                // Set then instrument on the channel.
                // Program is the voice.
                self.voices.insert(channel, program.as_int() as usize);
            },
            _ => ()
        }
    }

    fn gen_samples(&mut self, output_sample_rate: usize, output_channel_count: usize, output: &mut [f32]) -> usize {
        // Purge any finished notes.
        self.purge_finished_notes();

        tracing::debug!("Generating {} samples. {} Keys turned on.", output.len() / output_channel_count, self.keys.len());

        // Get the samples from the sampler bank for each note.
        for key in self.keys.iter_mut() {
            // Get the samples from the key.
            key.get_samples(output_channel_count, output);
        }

        output.len()
    }

    fn reset(&mut self) {
        self.keys = Vec::new();
        self.voices = HashMap::new();
    }
}