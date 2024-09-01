
import os
import sys
import librosa
import numpy as np

from PIL import Image
from typing import Optional

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.config import AudioDataTransformArgumentsConfig




class AudioTransform:
    def __init__(self, 
                 config: AudioDataTransformArgumentsConfig, 
                 audio_path: Optional[str] = None
                 ):
        
        super(AudioTransform, self).__init__()
        self.config = config
        self.audio_data = self.config

        self.audio_path = audio_path 
        self.target_sr = self.audio_data.target_sr # target_sr=16000
        self.top_db = self.audio_data.top_db # top_db=60
        self.scale = self.audio_data.scale # scale=True
        self.fix = self.audio_data.fix # fix=True
        self.mono = self.audio_data.mono # mono=True
        self.pad_mode = self.audio_data.pad_mode # pad_mode='reflect'


        self.frame_length = self.audio_data.frame_length # frame_length=512
        self.hop_length = self.audio_data.hop_length # hop_length=512

        self.n_steps = self.audio_data.n_steps # n_steps=4
        self.bins_per_octave = self.audio_data.bins_per_octave # bins_per_octave=32
        self.res_type = self.audio_data.res_type # res_type='kaiser_fast'
        self.rate = self.audio_data.rate # rate=12
        self.noise = self.audio_data.noise # noise=0.005




    # Sampling Rate Conversion
    # Ensure all audio files have the same sampling rate (e.g., 16kHz).

    def _resample_audio(self, audio_p) -> Optional[np.ndarray]:
        try:
            # Load audio file with its original sampling rate
            audio_f, sr = librosa.load(audio_p, sr=None)
            # Resample to target sampling rate
            audio_resampled = librosa.resample(audio_f, 
                                               orig_sr=sr, 
                                               target_sr=self.target_sr, 
                                               res_type='kaiser_fast', 
                                               scale=self.scale, 
                                               fix=self.fix,
                                               )
            return audio_resampled
        
        except Exception as e:
            logger.log_message("warning", "Failed to resample audio: " + str(e))
            my_exception = MyException(
                error_message = "Failed to resample audio: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    # Silence Removal
    # Remove silence from audio files to focus on the actual content.
    def _remove_silence(self, audio_f) -> Optional[np.ndarray]:
        try:
            # Remove silence
            y_no_silence, _ = librosa.effects.trim(audio_f, 
                                                   top_db=self.top_db, 
                                                   frame_length=self.frame_length,
                                                   hop_length=self.hop_length)
            
            return y_no_silence

        except Exception as e:
            logger.log_message("warning", "Failed to remove silence: " + str(e))
            my_exception = MyException(
                error_message = "Failed to remove silence: " + str(e),
                error_details = sys,
            )
            print(my_exception)

    #  Normalization
    # Normalize audio signals to adjust volume levels or apply z-score normalization.
    def _normalize_audio(self, audio_f) -> Optional[np.ndarray]:
        try:
            # Normalize audio to have zero mean and unit variance
            audio_f_normalized = (audio_f - np.mean(audio_f)) / np.std(audio_f)
            return audio_f_normalized

        except Exception as e:
            logger.log_message("warning", "Failed to normalize audio: " + str(e))
            my_exception = MyException(
                error_message = "Failed to normalize audio: " + str(e),
                error_details = sys,
            )
            print(my_exception)



    # Augmentation
    # Apply pitch shifting, time-stretching, or noise injection to increase data variability.

    def _augment_audio(self, audio_f) -> Optional[np.ndarray]:
        try:
            # Pitch shifting
            audio_f_shifted = librosa.effects.pitch_shift(audio_f, 
                                                          sr=self.target_sr, 
                                                          n_steps=self.n_steps, 
                                                          bins_per_octave=self.bins_per_octave,
                                                          res_type=self.res_type,
                                                          scale=self.scale,
                                                          )
            # Time-stretching
            audio_f_stretched = librosa.effects.time_stretch(audio_f, rate=self.rate)
            # Add noise
            noise = np.random.randn(len(audio_f))
            audio_f_noise = audio_f + self.noise * noise
            return audio_f_shifted, audio_f_stretched, audio_f_noise

        except Exception as e:
            logger.log_message("warning", "Failed to augment audio: " + str(e))
            my_exception = MyException(
                error_message = "Failed to augment audio: " + str(e),
                error_details = sys,
            )
            print(my_exception)


    def audio_processing(self) -> Optional[np.ndarray]:
        try: 
            audio_f = self._resample_audio(self.audio_path)
            audio_f_silence = self._remove_silence(audio_f)
            audio_f_normalized = self._normalize_audio(audio_f_silence)
            audio_f, audio_f_stretched, audio_f_noise = self._augment_audio(audio_f_normalized)
            # Concatenate all audio files
            audio_finally = np.concatenate((audio_f, audio_f_stretched, audio_f_noise), axis=0)

            return audio_finally

        except Exception as e:
            logger.log_message("warning", "Failed to process audio: " + str(e))
            my_exception = MyException(
                error_message = "Failed to process audio: " + str(e),
                error_details = sys,
            )
            print(my_exception)




        