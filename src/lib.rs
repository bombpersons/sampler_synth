use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::{Arc, Mutex};
use std::time::{Duration};
use std::{io, fs};
use std::path::{Path, PathBuf};
use std::fs::File;

use fundsp::hacker32::{envelope, wave32, declick, timer, constant};
use midly::MidiMessage;
use rubato::{SincFixedIn, InterpolationParameters, InterpolationType, ResamplerConstructionError, Resampler, ResampleError};
use serde::{Deserialize, Serialize};

use fundsp::wave::{Wave32};
use fundsp::prelude::{AudioUnit32, lerp};
use wav::{BitDepth};

use midi_player::Synth;

struct Channel {
    samples: Vec<f32>
}

#[derive(Debug)]
pub enum WavDataError {
    IoError(io::Error),
    ResamplerConstructionError(ResamplerConstructionError),
    ResampleError(ResampleError)
}

impl From<io::Error> for WavDataError {
    fn from(e: io::Error) -> Self {
        WavDataError::IoError(e)
    }
}

impl From<ResamplerConstructionError> for WavDataError {
    fn from(e: ResamplerConstructionError) -> Self {
        WavDataError::ResamplerConstructionError(e)
    }
}

impl From<ResampleError> for WavDataError {
    fn from(e: ResampleError) -> Self {
        WavDataError::ResampleError(e)
    }
}

pub fn wave32_from_file(filepath: &Path) -> Result<Wave32, WavDataError> {
    tracing::info!("Loading {}", filepath.display());

    // Open the file...
    let mut file = File::open(filepath)?;
    wave32_from_reader(&mut file)
}

pub fn wave32_from_reader<T: Read + Seek>(reader: &mut T) -> Result<Wave32, WavDataError> {
    let (header, data) = wav::read(reader)?;

    // Convert samples to f32
    let samples = match data {
        BitDepth::ThirtyTwoFloat(samples) => samples,
        BitDepth::TwentyFour(samples) => samples
            .iter().map(|s| ((*s as f32 + i32::MAX as f32) / (i32::MAX as f32 - i32::MIN as f32) - 0.5) * 2.0).collect(),
        BitDepth::Sixteen(samples) => samples
            .iter().map(|s| ((*s as f32 + i16::MAX as f32) / (i16::MAX as f32 - i16::MIN as f32) - 0.5) * 2.0).collect(),
        BitDepth::Eight(samples) => samples
            .iter().map(|s| ((*s as f32 / u8::MAX as f32) - 0.5) * 2.0).collect(),
        BitDepth::Empty => Vec::new()
    };

    // Create the wav32 to store the data.
    let sample_count = samples.len()/header.channel_count as usize;
    let mut wave32 = Wave32::with_capacity(header.channel_count as usize, header.sampling_rate as f64, sample_count);
    wave32.resize(sample_count);
    for (index, frame) in samples.chunks_exact(wave32.channels()).enumerate() {
        for (channel, sample) in frame.iter().enumerate() {
            wave32.set(channel, index, *sample);
        }
    }

    Ok(wave32)
}

#[derive(Debug)]
pub enum SampleError {
    WavDataError(WavDataError)
}

impl From<WavDataError> for SampleError {
    fn from(e: WavDataError) -> Self {
        SampleError::WavDataError(e)
    }
}

pub struct Sample {
    filepath: PathBuf,
    midi_note: usize,

    data: Option<Arc<Wave32>>
}

impl Sample {
    pub fn load(&mut self) -> Result<(), SampleError> {
        self.data = Some(Arc::new(wave32_from_file(&self.filepath)?));
        Ok(())
    }
}

#[derive(Debug)]
pub enum SamplerError {
    SampleError(SampleError),
    MismatchedSampleRates,
    NoSamplesFound,
}

impl From<SampleError> for SamplerError {
    fn from(e: SampleError) -> Self {
        SamplerError::SampleError(e)
    }
}

pub struct Sampler {
    name: String,
    // The wav data for the samples to use.
    // Multiple can be used, the nearest one 
    // to the desired note will be used.
    samples: Vec<Sample>,
}

impl Sampler {
    pub fn load_samples(&mut self) -> Result<(), SamplerError> {
        for sample in self.samples.iter_mut() {
            sample.load()?;
        }

        Ok(())
    }

    pub fn get_closest_sample(&self, midi_note: usize) -> Option<&Sample> {
        // Pick the sample with the closest midi note.
        let mut closest_sample = None;
        for sample in self.samples.iter() {
            closest_sample = match closest_sample {
                None => Some(sample),
                Some(closest) => {
                    if sample.midi_note.abs_diff(midi_note) < closest.midi_note.abs_diff(midi_note) {
                        Some(sample)
                    } else {
                        Some(closest)
                    }
                }
            }
        }
        closest_sample
    }
}

#[derive(Debug)]
pub enum SamplerBankError {
    IoError(io::Error),
    MalformedJson(serde_json::Error),
    SamplerError(SamplerError)
}

impl From<io::Error> for SamplerBankError {
    fn from(e: io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<serde_json::Error> for SamplerBankError {
    fn from(e: serde_json::Error) -> Self {
        Self::MalformedJson(e)
    }
}

impl From<SamplerError> for SamplerBankError {
    fn from(e: SamplerError) -> Self {
        Self::SamplerError(e)
    }
}

#[derive(Serialize, Deserialize)]
pub struct SamplerBankJson {
    voices: HashMap<String, Vec<(usize, usize)>>,
    folder: String,
}

pub struct SamplerBank {
    samplers: Vec<Sampler>,
    voices_to_samplers: HashMap<usize, usize>
}

impl SamplerBank {
    pub fn from_json_file(filepath: &Path) -> Result<Self, SamplerBankError> {
        // Try and open the file.
        let file = File::open(filepath)?;
        Self::from_json_reader(file)
    }

    pub fn from_json_reader<T: Read>(reader: T) -> Result<Self, SamplerBankError> {
        // Try and parse it.
        let parsed: SamplerBankJson = serde_json::from_reader(reader)?;

        let mut bank = Self {
            samplers: Vec::new(),
            voices_to_samplers: HashMap::new()
        };

        // List all of the paths in the specified folder.
        let paths = fs::read_dir(&parsed.folder)?;

        // Iterate over each directory and populate our samplers.
        for sampler_dir_result in paths {
            let sampler_dir = sampler_dir_result?;

            // Check if this item is actually a directory.
            match sampler_dir.metadata() {
                Ok(metadata) => if metadata.is_file() { continue },
                Err(e) => tracing::warn!("Couldn't parse {} due to error: {}", sampler_dir.path().display(), e)
            }

            // Figure out the name for the sampler (just use the directory name.)
            let sampler_name = match sampler_dir.path().file_stem() {
                None => {
                    tracing::warn!("Couldn't get sampler directory name. Skipping.");
                    continue;
                },
                Some(name) => {
                    name.to_string_lossy().to_string()
                }
            };
            
            // Create the sampler.
            let mut sampler = Sampler {
                name: sampler_name,
                samples: Vec::new()
            };

            // Look at the file structure and add the individual samples to the sampler.
            let samples = fs::read_dir(sampler_dir.path())?;
            for sample_file_result in samples {
                let sample_file = sample_file_result?;

                // Only look at files that have a wav extension.
                match sample_file.path().extension() {
                    None => continue, // No extension.
                    Some(ext) => if ext != "wav" { continue; }
                }

                // Figure out the name for the sample.
                let sample_name = match sample_file.path().file_stem() {
                    None => {
                        tracing::warn!("Couldn't get sampler directory name. Skipping.");
                        continue;
                    },
                    Some(name) => {
                        name.to_string_lossy().to_string()
                    }
                };

                // Calculate the midi note fromthe file name.
                let midi_note = match midi_player::note_name_to_midi_note(&sample_name) {
                    Err(e) => {
                        tracing::warn!("Error parsing sample as note name. {}", e);
                        continue;
                    },
                    Ok(note) => note
                };

                // Create the sample and add it to the sampler.
                let sample = Sample {
                    data: None,
                    filepath: sample_file.path(),
                    midi_note: midi_note as usize
                };
                sampler.samples.push(sample);
            }
            bank.samplers.push(sampler);
        }

        // Build a mapping of midi instrument code to sampler index.
        for (voice_name, codes) in parsed.voices.iter() {
            // Find the voice in the sampler list.
            let sampler = bank.get_sampler_by_name(voice_name);

            // If we found it, go over the ranges that the voices is applicable for.
            if let Some((sampler_index, _)) = sampler {
                for range in codes.iter() {
                    // Range is inclusive.
                    for i in range.0..range.1+1 { 
                        bank.voices_to_samplers.insert(i, sampler_index);
                    }   
                }
            }
        }

        Ok(bank)
    }

    pub fn get_sampler_by_midi_instrument(&self, index: usize) -> Option<&Sampler> {
        let sampler_index = self.voices_to_samplers.get(&index)?;
        self.samplers.get(*sampler_index)
    }

    pub fn get_sampler_by_name(&self, name: &str) -> Option<(usize, &Sampler)> {
        for (index, sampler) in self.samplers.iter().enumerate() {
            if sampler.name == name {
                return Some((index, sampler));
            }
        }
        None
    }

    pub fn load_samplers(&mut self) -> Result<(), SamplerBankError> {
        for sampler in self.samplers.iter_mut() {
            sampler.load_samples()?;
        }
        Ok(())
    }
}

pub struct SamplerVoice {
    channel: usize,
    midi_note: usize,
    vel: f32,

    // The audio node that plays our sample.
    audio_node: Box<dyn AudioUnit32>,

    // So that we can tell the audio node when we have released the key.
    time_released: Arc<Mutex<Option<f32>>>,

    // The total duration of the sample. So we know when we've finished playing.
    sample_duration: f32
}

#[derive(Debug)]
pub enum SampleVoiceError {
    NoSampleFound,
    SampleNotLoaded
}

// Tag the time in the audio graph so that we can get access it later when the sample is playing.
const SAMPLERVOICE_TIMER_TAG: i64 = 0;
const SAMPLERVOICE_FADE_TIME: f32 = 0.1;
impl SamplerVoice {
    pub fn from_sampler(sampler: &Sampler, sample_rate: usize, channel: usize, midi_note: usize, vel: f32) -> Result<SamplerVoice, SampleVoiceError> {
        // Create a fade that so when a key is released, the sample fades out rather than
        // stopping abruptly.
        let time_released = Arc::new(Mutex::new(None));
        let volume_modulation = {
            let time_released = time_released.clone();
            envelope(move |t| {
                let time_released = time_released.lock().unwrap();
                match *time_released {
                    None => 1.0,
                    Some(released) => lerp(1.0, 0.0, ((released - t) as f32 / SAMPLERVOICE_FADE_TIME).min(1.0))
                }
            })
        };

        // Get the sample data.
        let sample = sampler.get_closest_sample(midi_note)
            .ok_or(SampleVoiceError::NoSampleFound)?;
        let sample_data = sample.data.as_ref()
            .ok_or(SampleVoiceError::SampleNotLoaded)?;

        // Create an audio node that can play the sample.
        let mut audio_node = 
            timer(SAMPLERVOICE_TIMER_TAG) |
            (
                wave32(sample_data.clone(), 0, None) >> declick() * constant(vel) |
                wave32(sample_data.clone(), 1, None) >> declick() * constant(vel)
            );
        audio_node.reset(Some(sample_rate as f64));

        // Create the voice and return it.
        let voice = Self {
            channel,
            midi_note,
            vel,
            audio_node: Box::new(audio_node),
            time_released,
            sample_duration: sample_data.duration() as f32
        };
        Ok(voice)
    }

    pub fn get_time_played(&self) -> f32 {
        self.audio_node.get(SAMPLERVOICE_TIMER_TAG)
            .expect("The time tag should always be present!") as f32
    }

    pub fn set_released(&mut self) {
        let mut time_released = self.time_released.lock().unwrap();
        let current_time = self.get_time_played();
        *time_released = Some(current_time as f32);
    }

    pub fn is_pressed(&self) -> bool {
        let mut time_released = self.time_released.lock().unwrap();
        time_released.is_none() 
    }

    pub fn is_finished(&self) -> bool {
        let time_released = self.time_released.lock().unwrap();
        let current_time = self.get_time_played();
        
        current_time > self.sample_duration ||
        match *time_released {
            None => false,
            Some(time_released) => current_time > time_released + SAMPLERVOICE_FADE_TIME
        }
    }

    pub fn get_samples(&mut self, output_channel_count: usize, output: &mut [f32]) {
        const MAX_CHANNEL_COUNT: usize = 2;

        let freq = midi_player::midi_note_to_freq(self.midi_note as u8);
        let inputs = [freq];

        let voice_output_count = self.audio_node.outputs();
        if voice_output_count > MAX_CHANNEL_COUNT {
            panic!("Voice has too many outputs / channels!");
        }

        // Generate samples for each of these buffers.
        let mut temp_buf = [0.0; MAX_CHANNEL_COUNT];
        for frame in output.chunks_exact_mut(output_channel_count) {
            // The output from the audio node.
            self.audio_node.tick(&inputs, &mut temp_buf[..voice_output_count]);

            for channel in 0..output_channel_count {
                if channel >= voice_output_count {
                    frame[channel] += temp_buf[voice_output_count-1];
                } else {
                    frame[channel] += temp_buf[channel];
                }
            }
        }
    } 
}

pub struct SamplerSynth {
    bank: SamplerBank,
    voices: Vec<SamplerVoice>,
    sample_rate: usize,

    // Map of channels to midi voices.
    midi_instrument_to_sample_index: HashMap<usize, usize>
}

impl SamplerSynth {
    pub fn new(bank: SamplerBank) -> Self {
        Self {
            bank,
            voices: Vec::new(),
            sample_rate: 0, // Get's set the first time gen_samples() is called.

            midi_instrument_to_sample_index: HashMap::new()
        }
    }

    fn note_on(&mut self, channel: usize, midi_note: usize, vel: usize) {
        tracing::debug!("Key {} on at {} velocity", midi_note, vel);

        // Find the correct sampler.
        let instrument_code = match self.midi_instrument_to_sample_index.get(&channel) {
            None => {
                tracing::warn!("No voice for channel {} found", channel);
                return;
            },
            Some(code) => code
        };

        let sampler = match self.bank.get_sampler_by_midi_instrument(*instrument_code) {
            None => {
                tracing::warn!("No sampler found for instrument {} on channel {}", instrument_code, channel);
                return;
            }
            Some(sampler) => sampler
        };

        // Add the voice
        let voice = SamplerVoice::from_sampler(sampler,
            self.sample_rate,
            channel,
            midi_note, 
            vel as f32 / 127.0
        ).expect("Error creating SamplerVoice.");
        self.voices.push(voice);
    }

    fn note_off(&mut self, channel: usize, midi_note: usize) {
        tracing::debug!("Key {} off", midi_note);

        // Find the note and record the time it stopped.
        for key in self.voices.iter_mut() {
            if key.channel == channel && 
               key.midi_note == midi_note && 
               key.is_pressed() {

                key.set_released();
            }
        }
    }

    fn purge_finished_notes(&mut self, threshold: usize) {
        // Get rid of any notes that have played more than the threshold of samples past where they stopped.
        self.voices.retain(|key| {
            !key.is_finished()
        });
    }
}

impl Synth for SamplerSynth {
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
                self.midi_instrument_to_sample_index.insert(channel, program.as_int() as usize);
            },
            _ => ()
        }
    }

    fn gen_samples(&mut self, output_sample_rate: usize, output_channel_count: usize, output: &mut [f32]) -> usize {
        // Store the sample rate requested.
        self.sample_rate = output_sample_rate;

        // Purge any finished notes.
        self.purge_finished_notes(output_sample_rate); // 1 second.

        tracing::debug!("Generating {} samples. {} Keys turned on.", output.len() / output_channel_count, self.voices.len());

        // Get the samples from the sampler bank for each note.
        for key in self.voices.iter_mut() {
            key.get_samples(output_channel_count, output);
        }

        output.len()
    }

    fn reset(&mut self) {
        self.voices = Vec::new();
        self.midi_instrument_to_sample_index = HashMap::new();
    }
}