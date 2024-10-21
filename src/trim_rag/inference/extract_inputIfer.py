import os
import sys
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union

from src.trim_rag.logger import logger
from src.trim_rag.exception import MyException
from src.trim_rag.processing import TextTransform, ImageTransform, AudioTransform


class TextDataInference(TextTransform):
    def __init__(self, **kwargs) -> None:
        super(TextDataInference, self).__init__(**kwargs)

    def text_processing(self, input: str) -> Optional[str]:
        try:
            logger.log_message(
                "info", "Pre-Processing text data in inference phase started."
            )
            text = self._extract_text_from_pdf(pdf_path=input)
            text = self._normalize_text(text)
            text = self._remove_stopwords(text)
            text = self._lemmatize_text(text)

            logger.log_message(
                "info",
                "Pre-Processing text data in inference phase completed successfully.",
            )
            return text

        except Exception as e:
            logger.log_message(
                "warning", "Failed to run text processing in Inference Phase: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to run text processing in Inference Phase: "
                + str(e),
                error_details=sys,
            )
            print(my_exception)


class ImageDataInference(ImageTransform):
    def __init__(self, **kwargs) -> None:
        super(ImageDataInference, self).__init__(**kwargs)

    def image_processing(self, input):
        try:
            logger.log_message(
                "info", "Pre-Processing image data in inference phase started."
            )
            image = Image.open(input)
            resized_image = image.resize((self.size, self.size))
            augmented_image = self._augment_image(resized_image)
            converted_image = self._convert_format(augmented_image)

            logger.log_message(
                "info",
                "Pre-Processing image data in inference phase completed successfully.",
            )

            return converted_image

        except Exception as e:
            logger.log_message(
                "warning", "Failed to process image in inference phase: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to process image in inference phase: " + str(e),
                error_details=sys,
            )
            print(my_exception)


class AudioDataInference(AudioTransform):
    def __init__(self, **kwargs) -> None:
        super(AudioDataInference, self).__init__(**kwargs)

    def audio_processing(self, input):
        try:
            logger.log_message(
                "info", "Pre-Processing audio data in inference phase started."
            )
            # PLit audio longer than 30 minutes
            self._plit_and_create(audio_p=input)
            audio_f = self._resample_audio(audio_p=input)
            audio_f_silence = self._remove_silence(audio_f)
            audio_f_normalized = self._normalize_audio(audio_f_silence)
            audio_f, audio_f_stretched, audio_f_noise = self._augment_audio(
                audio_f_normalized
            )
            # Concatenate all audio files
            audio_finally = np.concatenate(
                (audio_f, audio_f_stretched, audio_f_noise), axis=0
            )

            logger.log_message(
                "info",
                "Pre-Processing audio data in inference phase completed successfully.",
            )

            return audio_finally

        except Exception as e:
            logger.log_message(
                "warning", "Failed to process audio in inference phase: " + str(e)
            )
            my_exception = MyException(
                error_message="Failed to process audio in inference phase: " + str(e),
                error_details=sys,
            )
            print(my_exception)
