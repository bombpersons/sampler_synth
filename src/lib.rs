use std::collections::HashMap;
use std::io::{Read, Seek};
use std::time::{Duration};
use std::{io, fs};
use std::io::{Write};
use std::path::{Path, PathBuf};
use std::fs::File;

use midly::MidiMessage;
use rubato::{SincFixedIn, InterpolationParameters, InterpolationType, ResamplerConstructionError, Resampler, ResampleError};
use serde::{Deserialize, Serialize};
use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;

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

struct WavData {
    sample_rate: u16,
    channels: Vec<Channel>,
    sample_length: usize
}

impl WavData {
    pub fn from_file(filepath: &Path) -> Result<Self, WavDataError> {
        tracing::info!("Loading {}", filepath.display());

        // Open the file...
        let mut file = File::open(filepath)?;
        Self::from_reader(&mut file)
    }

    pub fn from_reader<T: Read + Seek>(reader: &mut T) -> Result<Self, WavDataError> {
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

        // Format the data. Separate the channels into vectors
        // for easy access. Plus our resampler needs them like this.
        let mut channels = Vec::new();
        for _ in 0..header.channel_count {
            channels.push(Channel {
                samples: Vec::new()
            });
        }
        for frame in samples.chunks_exact(header.channel_count as usize) {
            for (i, s) in frame.iter().enumerate() {
                channels[i].samples.push(*s);
            }
        }

        // Record the length of the sample. All channels *should* be the same length.
        let sample_length = if channels.len() > 0 {
            channels[0].samples.len()
        } else {
            0
        };

        let data = WavData {
            sample_rate: header.sampling_rate as u16,
            channels,
            sample_length
        };
        Ok(data)
    }

    pub fn calculate_pitch(&self) -> Option<f32> {
        const POWER_THRESHOLD: f32 = 5.0;
        const CLARITY_THRESHOLD: f32 = 0.7;

        let mut pitches = Vec::new();
        for channel in self.channels.iter() {
            let size = channel.samples.len();
            let padding = size*2;

            let mut detector = McLeodDetector::new(channel.samples.len(), padding);
            match detector.get_pitch(&channel.samples, self.sample_rate as usize, POWER_THRESHOLD, CLARITY_THRESHOLD)
            {
                None => (),
                Some(pitch) => pitches.push(pitch.frequency)
            }
        }

        if pitches.is_empty() {
            None
        } else {
            Some(pitches.iter().sum::<f32>() / pitches.len() as f32)
        }
    }

    pub fn resample(&self, new_sample_rate: u16) -> Result<Self, WavDataError> {
        tracing::info!("Resampling from {} samples per second to {} samples per second...", self.sample_rate, new_sample_rate);

        // Create the resampler...
        // I have no idea what these options really do, just using the ones used on the readme.
        let params = InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: InterpolationType::Linear,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2
        };
        let mut resampler = SincFixedIn::<f32>::new(
            new_sample_rate as f64 /  self.sample_rate as f64,
            2.0,
            params,
            self.sample_length,
            self.channels.len()
        )?;

        // Construct the vectors required for rubato
        let mut waves_in = Vec::new();
        for channel in self.channels.iter() {
            waves_in.push(channel.samples.to_vec());
        }

        // Process
        let waves_out = resampler.process(&waves_in, None)?;

        // Put the data back into channels
        let mut channels = Vec::new();
        for c in waves_out.iter() {
            channels.push(Channel { samples: c.to_vec() });
        };

        // Construct the new wavdata.
        let new = Self {
            channels,
            sample_length: self.sample_length,
            sample_rate: new_sample_rate
        };

        tracing::info!("Resampled.");

        Ok(new)
    }
}

#[derive(Debug)]
pub enum SampleError {
    SampleNotLoaded,
    ChannelOutOfBounds,
    SampleOutOfBounds,
    WavDataError(WavDataError)
}

impl From<WavDataError> for SampleError {
    fn from(e: WavDataError) -> Self {
        SampleError::WavDataError(e)
    }
}

#[derive(Serialize, Deserialize)]
struct Sample {
    filepath: PathBuf,
    pitch: Option<f32>,

    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    data: Option<WavData>
}

impl Sample {
    pub fn load(&mut self) -> Result<(), SampleError> {
        let wav_data = WavData::from_file(&self.filepath)?;

        // Detect the pitch of the sample.
        if self.pitch.is_none() {
            tracing::info!("Autodetecting pitch for {}.", self.filepath.display());
            self.pitch = wav_data.calculate_pitch();
            match self.pitch {
                Some(pitch) => tracing::info!("Frequency detected to be {} for {}.", pitch, self.filepath.display()),
                None => tracing::warn!("Pitch couldn't be detected for {}.", self.filepath.display())
            };
        }

        self.data = Some(wav_data);

        Ok(())
    }

    pub fn resample(&mut self, new_sample_rate: u16) -> Result<(), SampleError> {
        if self.data.is_none() {
            return Err(SampleError::SampleNotLoaded);
        }

        self.data = Some(self.data.as_ref().unwrap().resample(new_sample_rate)?);
        Ok(())
    }

    pub fn get_channel(&self, channel: usize) -> Result<&Channel, SampleError> {
        if self.data.is_none() {
            return Err(SampleError::SampleNotLoaded);
        }

        let channel = self.data.as_ref().unwrap().channels.get(channel);
        match channel {
            Some(c) => Ok(c),
            None => Err(SampleError::ChannelOutOfBounds)
        }
    }

    pub fn get_sample_rate(&self) -> Result<u16, SampleError> {
        if self.data.is_none() {
            return Err(SampleError::SampleNotLoaded);
        }
        Ok(self.data.as_ref().unwrap().sample_rate)
    }

    pub fn get_sample_channel_count(&self) -> Result<usize, SampleError> {
        if self.data.is_none() {
            return Err(SampleError::SampleNotLoaded);
        }
        Ok(self.data.as_ref().unwrap().channels.len())
    }

    pub fn get_sample_interpolated(&self, index: f32, channel: usize) -> Result<f32, SampleError> {
        let channel = self.get_channel(channel)?;

        let low_sample_index = index.floor() as usize;
        let high_sample_index = index.ceil() as usize;

        // If we have no low sample error
        match channel.samples.get(low_sample_index) {
            None => Err(SampleError::SampleOutOfBounds),
            Some(low_sample) => {
                // Try and get the high sample and interpolate. If it doesn't exist, just return then low one.
                match channel.samples.get(high_sample_index) {
                    Some(high_sample) => {
                        // Interpolate between both samples.
                        let remainder = index - low_sample_index as f32;
                        Ok(low_sample + remainder * (high_sample - low_sample))
                    },
                    _ => Ok(*low_sample),
                }
            }
        }
    }

    pub fn get_pitch(&self) -> f32 {
        match self.pitch {
            Some(pitch) => pitch,
            None => 440.0
        }
    }
    
    pub fn get_samples(&self, output_sample_rate: usize, output_channels: usize, desired_pitch: f32, volume: f32, mut progress: usize, samples_stopped_at: Option<usize>, output: &mut [f32]) -> Result<usize, SampleError> {
        // Ratio of output samples per actual samples.
        let sample_rate = self.get_sample_rate()?;
        let output_sample_num = output.len() / output_channels;
        let samples_per_output_sample = sample_rate as f32 / output_sample_rate as f32;

        // How many channels are in this sample.
        let sample_channels = self.get_sample_channel_count()?;

        // How much faster do we need to sample in order
        // to get the desired frequency?
        let desired_freq = desired_pitch;
        let sample_freq = self.get_pitch();
        let freq_ratio = desired_freq / sample_freq;

        // When do we stop sampling?
        let mut sampled = 0;
        while sampled < output_sample_num {
            // Calculate the sample index we need to be getting right now.
            let sample_index = progress as f32 * samples_per_output_sample as f32 * freq_ratio;
            let progress_as_time = Duration::from_secs_f32(progress as f32 / output_sample_rate as f32);

            //tracing::info!("Sample_index: {}", sample_index);

            // Fill out each channel.
            for channel in 0..output_channels {
                let channel_to_sample = (sample_channels-1).min(channel);
                let sample = match self.get_sample_interpolated(sample_index, channel_to_sample) {
                    Err(SampleError::SampleOutOfBounds) => { // Sample out of bounds, return 0.0
                        Ok(0.0)
                    },
                    Ok(mut sample) => {
                        // Fade in the first moment of the sample to avoid clipping.
                        const FADE_IN_DURATION: f32 = 0.01;
                        let fade_in = (progress_as_time.as_secs_f32() / FADE_IN_DURATION).min(1.0);
                        sample *= fade_in;

                        // Fade out in the last moment of the sample to avoid clipping.
                        const FADE_OUT_DURATION: f32 = 0.1;
                        if let Some(samples_stopped_at) = samples_stopped_at {
                            let time_stopped_at = Duration::from_secs_f32(samples_stopped_at as f32 / output_sample_rate as f32);
                            let duration_since_stopped = (progress_as_time - time_stopped_at).max(Duration::ZERO);
                            let fade_out = 1.0 - (duration_since_stopped.as_secs_f32() / FADE_OUT_DURATION).min(1.0);

                            sample *= fade_out;
                        }

                        // Volume
                        sample *= volume;

                        Ok(sample)
                    },
                    Err(e) => Err(e), // Return the error unmodified
                }?;
                output[sampled*output_channels + channel] += sample;
            }

            progress += 1;
            sampled += 1;
        }

        Ok(output_sample_num)
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

#[derive(Serialize, Deserialize)]
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

    // Resample all samples to a new sample rate.
    pub fn resample(&mut self, new_sample_rate: u16) -> Result<(), SamplerError> {
        for sample in self.samples.iter_mut() {
            sample.resample(new_sample_rate)?;
        }
        Ok(())
    }

    // Retrieve the samples for a particular note. Returns the number of samples returned.
    pub fn get_samples(&self, output_sample_rate: usize, output_channels: usize, pitch: f32, volume: f32, progress: usize, time_stopped: Option<usize>, output: &mut [f32]) -> Result<usize, SamplerError> {
        // Pick the sample with the closest midi note.
        let mut closest_sample = None;
        for sample in self.samples.iter() {
            closest_sample = match closest_sample {
                None => Some(sample),
                Some(closest) => {
                    if (sample.get_pitch() - pitch).abs() < (closest.get_pitch() - pitch).abs() {
                        Some(sample)
                    } else {
                        Some(closest)
                    }
                }
            }
        }
        if closest_sample.is_none() {
            return Err(SamplerError::NoSamplesFound);
        }

        let sample = closest_sample.unwrap();
        let sampled = 
            sample.get_samples(output_sample_rate, 
                output_channels, 
                pitch, 
                volume,
                progress, 
                time_stopped, 
                output)?;
        Ok(sampled)
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
pub struct SamplerBank {
    voices: HashMap<String, Vec<(usize, usize)>>,
    folder: String,
    samplers: Vec<Sampler>,

    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    voices_to_samplers: HashMap<usize, usize>
}

impl SamplerBank {
    pub fn to_json_file(&self, filepath: &Path) -> Result<(), SamplerBankError> {
        // Try and open the file.
        let file = File::create(filepath)?;
        self.to_json_writer(file);

        Ok(())
    }

    pub fn to_json_writer<T: Write>(&self, mut writer: T) -> Result<(), SamplerBankError> {
        let json_string = serde_json::to_string_pretty(&self)?;
        writer.write(json_string.as_bytes());

        Ok(())
    }

    pub fn from_json_file(filepath: &Path) -> Result<Self, SamplerBankError> {
        // Try and open the file.
        let file = File::open(filepath)?;
        Self::from_json_reader(file)
    }

    pub fn from_json_reader<T: Read>(reader: T) -> Result<Self, SamplerBankError> {
        // Try and parse it.
        let mut parsed: SamplerBank = serde_json::from_reader(reader)?;

        // Automatically fill the samplers out using the directory specified.
        // If there are samplers already, don't do this.
        if parsed.samplers.is_empty() {
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
                    let pitch = match midi_player::note_name_to_midi_note(&sample_name) {
                        Err(e) => {
                            tracing::info!("Error parsing note name from sample filename. Will use auto detection. Error: {}", e);
                            None
                        },
                        Ok(note) => Some(midi_player::midi_note_to_freq(note))
                    };

                    // Create the sample and add it to the sampler.
                    let sample = Sample {
                        data: None,
                        filepath: sample_file.path(),
                        pitch
                    };
                    sampler.samples.push(sample);
                }
                parsed.samplers.push(sampler);
            }
        }

        // Build a mapping of midi instrument code to sampler index.
        for (voice_name, codes) in parsed.voices.iter() {
            // Find the voice in the sampler list.
            let sampler = parsed.get_sampler_by_name(voice_name);

            // If we found it, go over the ranges that the voices is applicable for.
            if let Some((sampler_index, _)) = sampler {
                for range in codes.iter() {
                    // Range is inclusive.
                    for i in range.0..range.1+1 { 
                        parsed.voices_to_samplers.insert(i, sampler_index);
                    }   
                }
            }
        }

        Ok(parsed)
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

    pub fn resample(&mut self, new_sample_rate: u16) -> Result<(), SamplerBankError> {
        for sampler in self.samplers.iter_mut() {
            sampler.resample(new_sample_rate)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Key {
    channel: usize,
    midi_note: usize,
    vel: f32,
    samples_played: usize,
    samples_stopped_at: Option<usize>
}

pub struct SamplerSynth {
    bank: SamplerBank,
    keys: Vec<Key>,

    // Map of channels to midi voices.
    voices: HashMap<usize, usize>
}

impl SamplerSynth {
    pub fn new(bank: SamplerBank) -> Self {
        Self {
            bank,
            keys: Vec::new(),
            voices: HashMap::new()
        }
    }

    fn note_on(&mut self, channel: usize, midi_note: usize, vel: usize) {
        tracing::debug!("Key {} on at {} velocity", midi_note, vel);

        // Add the note.
        self.keys.push(Key {
            channel,
            midi_note, 
            vel: vel as f32 / 127.0,
            samples_played: 0,
            samples_stopped_at: None
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
            }
        }
    }

    fn purge_finished_notes(&mut self, threshold: usize) {
        // Get rid of any notes that have played more than the threshold of samples past where they stopped.
        self.keys.retain(|key| {
            if let Some(stopped_at) = key.samples_stopped_at {
                key.samples_played - stopped_at <= threshold
            } else {
                true
            }   
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
                self.voices.insert(channel, program.as_int() as usize);
            },
            _ => ()
        }
    }

    fn gen_samples(&mut self, output_sample_rate: usize, output_channel_count: usize, output: &mut [f32]) -> usize {
        // Purge any finished notes.
        self.purge_finished_notes(output_sample_rate); // 1 second.

        tracing::debug!("Generating {} samples. {} Keys turned on.", output.len() / output_channel_count, self.keys.len());

        // Get the samples from the sampler bank for each note.
        for key in self.keys.iter_mut() {
            // Find the right sampler for the key...
            let instrument_code = match self.voices.get(&key.channel) {
                None => {
                    tracing::warn!("No voice for channel {} found", key.channel);
                    continue;
                },
                Some(code) => code
            };

            // Try to get the sampler and generate samples using it.
            match self.bank.get_sampler_by_midi_instrument(*instrument_code) {
                None => tracing::warn!("No sampler found for instrument {} on channel {}", instrument_code, key.channel),
                Some(sampler) => {
                    // Generate samples.
                    let samples_generated = sampler.get_samples(output_sample_rate, 
                        output_channel_count, 
                        midi_player::midi_note_to_freq(key.midi_note as u8), 
                        key.vel, 
                        key.samples_played, 
                        key.samples_stopped_at, 
                        output).unwrap();
        
                    key.samples_played += samples_generated;
                }
            };
        }

        output.len()
    }

    fn reset(&mut self) {
        self.keys = Vec::new();
        self.voices = HashMap::new();
    }
}