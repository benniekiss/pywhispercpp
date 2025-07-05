import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

import pywhispercpp.constants as constants
from pywhispercpp.lib import _pywhispercpp as pw
from pywhispercpp.logging import suppress_stdout_stderr

logger = logging.getLogger(__name__)


class Segment:
    """
    A small class representing a transcription segment
    """

    def __init__(self, t0: int, t1: int, text: str):
        """
        :param t0: start time
        :param t1: end time
        :param text: text
        """
        self.t0 = t0
        self.t1 = t1
        self.text = text

    def __str__(self):
        return f"t0={self.t0}, t1={self.t1}, text={self.text}"

    def __repr__(self):
        return str(self)


class Model:
    """
    This classes defines a Whisper.cpp model.

    Example usage.
    ```python
    model = Model('base.en', n_threads=6)
    segments = model.transcribe('file.mp3')
    for segment in segments:
        print(segment.text)
    ```
    """

    _new_segment_callback = None

    def __init__(
        self,
        model: str = "tiny",
        models_dir: str | None = None,
        params_sampling_strategy: int = 0,
        verbose: bool = True,
        **params,
    ):
        """
        :param model: The name of the model, one of the [AVAILABLE_MODELS](/pywhispercpp/#pywhispercpp.constants.AVAILABLE_MODELS),
                        (default to `tiny`), or a direct path to a `ggml` model.
        :param models_dir: The directory where the models are stored, or where they will be downloaded if they don't
                            exist, default to [MODELS_DIR](/pywhispercpp/#pywhispercpp.constants.MODELS_DIR) <user_data_dir/pywhsipercpp/models>
        :param params_sampling_strategy: 0 -> GREEDY, else BEAM_SEARCH
        :param redirect_whispercpp_logs_to: where to redirect the whisper.cpp logs, default to False (no redirection), accepts str file path, sys.stdout, sys.stderr, or use None to redirect to devnull
        :param params: keyword arguments for different whisper.cpp parameters,
                        see [PARAMS_SCHEMA](/pywhispercpp/#pywhispercpp.constants.PARAMS_SCHEMA)
        """
        if not Path(model).is_file():
            raise FileNotFoundError(f"Model, {model}, could not be found on disk.")

        self.model_path = model

        self._ctx = None
        self._sampling_strategy = (
            pw.whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
            if params_sampling_strategy == 0
            else pw.whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH
        )
        self._params = pw.whisper_full_default_params(self._sampling_strategy)
        # assign params
        self._set_params(params)
        self.verbose = verbose
        # init the model
        self._init_model()

    def transcribe(
        self,
        media: np.ndarray,
        n_processors: int | None = None,
        new_segment_callback: Callable[[Segment], None] | None = None,
        **params,
    ) -> list[Segment]:
        """
        Transcribes the media provided as input and returns list of `Segment` objects.
        Accepts a media_file path (audio/video) or a raw numpy array.

        :param media: Media file path or a numpy array
        :param n_processors: if not None, it will run the transcription on multiple processes
                             binding to whisper.cpp/whisper_full_parallel
                             > Split the input audio in chunks and process each chunk separately using whisper_full()
        :param new_segment_callback: callback function that will be called when a new segment is generated
        :param params: keyword arguments for different whisper.cpp parameters, see ::: constants.PARAMS_SCHEMA

        :return: List of transcription segments
        """
        if not isinstance(media, np.ndarray):
            raise TypeError("media must be a 1D np.ndarray")

        # update params if any
        self._set_params(params)

        # setting up callback
        if new_segment_callback:
            Model._new_segment_callback = new_segment_callback
            pw.assign_new_segment_callback(
                self._params, Model.__call_new_segment_callback
            )

        res = self._transcribe(media, n_processors=n_processors)

        return res

    @staticmethod
    def _get_segments(
        ctx,
        start: int,
        end: int,
    ) -> list[Segment]:
        """
        Helper function to get generated segments between `start` and `end`

        :param start: start index
        :param end: end index

        :return: list of segments
        """
        n = pw.whisper_full_n_segments(ctx)
        assert end <= n, (
            f"{end} > {n}: `End` index must be less or equal than the total number of segments"
        )
        res = []
        for i in range(start, end):
            t0 = pw.whisper_full_get_segment_t0(ctx, i)
            t1 = pw.whisper_full_get_segment_t1(ctx, i)
            bytes = pw.whisper_full_get_segment_text(ctx, i)
            text = bytes.decode("utf-8", errors="replace")
            res.append(Segment(t0, t1, text.strip()))
        return res

    def get_params(self) -> dict:
        """
        Returns a `dict` representation of the actual params

        :return: params dict
        """
        res = {}
        for param in dir(self._params):
            if param.startswith("__"):
                continue
            try:
                res[param] = getattr(self._params, param)
            except Exception:
                # ignore callback functions
                continue
        return res

    @staticmethod
    def get_params_schema() -> dict:
        """
        A simple link to ::: constants.PARAMS_SCHEMA
        :return: dict of params schema
        """
        return constants.PARAMS_SCHEMA

    @staticmethod
    def lang_max_id() -> int:
        """
        Returns number of supported languages.
        Direct binding to whisper.cpp/lang_max_id
        :return:
        """
        return pw.whisper_lang_max_id()

    def print_timings(self) -> None:
        """
        Direct binding to whisper.cpp/whisper_print_timings

        :return: None
        """
        pw.whisper_print_timings(self._ctx)

    @staticmethod
    def system_info() -> None:
        """
        Direct binding to whisper.cpp/whisper_print_system_info

        :return: None
        """
        return pw.whisper_print_system_info()

    @staticmethod
    def available_languages() -> list[str]:
        """
        Returns a list of supported language codes

        :return: list of supported language codes
        """
        n = pw.whisper_lang_max_id()
        res = []
        for i in range(n):
            res.append(pw.whisper_lang_str(i))
        return res

    def _init_model(self) -> None:
        """
        Private method to initialize the method from the bindings, it will be called automatically from the __init__
        :return:
        """
        logger.info("Initializing the model ...")

        disable = not self.verbose
        with suppress_stdout_stderr(disable=disable):
            self._ctx = pw.whisper_init_from_file_with_params(self.model_path)

    def _set_params(self, kwargs: dict) -> None:
        """
        Private method to set the kwargs params to the `Params` class
        :param kwargs: dict like object for the different params
        :return: None
        """
        for param in kwargs:
            setattr(self._params, param, kwargs[param])

    def _transcribe(
        self,
        audio: np.ndarray,
        n_processors: int | None = None,
    ):
        """
        Private method to call the whisper.cpp/whisper_full function

        :param audio: numpy array of audio data
        :param n_processors: if not None, it will run whisper.cpp/whisper_full_parallel with n_processors
        :return:
        """

        if n_processors:
            pw.whisper_full_parallel(
                self._ctx, self._params, audio, audio.size, n_processors
            )
        else:
            pw.whisper_full(self._ctx, self._params, audio, audio.size)
        n = pw.whisper_full_n_segments(self._ctx)
        res = Model._get_segments(self._ctx, 0, n)
        return res

    @staticmethod
    def __call_new_segment_callback(
        ctx,
        n_new,
        user_data,
    ) -> None:
        """
        Internal new_segment_callback, it just calls the user's callback with the `Segment` object
        :param ctx: whisper.cpp ctx param
        :param n_new: whisper.cpp n_new param
        :param user_data: whisper.cpp user_data param
        :return: None
        """
        n = pw.whisper_full_n_segments(ctx)
        start = n - n_new
        res = Model._get_segments(ctx, start, n)
        for segment in res:
            Model._new_segment_callback(segment)

    def auto_detect_language(
        self,
        media: np.ndarray,
        offset_ms: int = 0,
        n_threads: int = 4,
    ) -> tuple[tuple[str, np.float32], dict[str, np.float32]]:
        """
        Automatic language detection using whisper.cpp/whisper_pcm_to_mel and whisper.cpp/whisper_lang_auto_detect

        :param media: Media file path or a numpy array
        :param offset_ms: offset in milliseconds
        :param n_threads: number of threads to use
        :return: ((detected_language, probability), probabilities for all languages)
        """
        if not isinstance(media, np.ndarray):
            raise TypeError("media must be a 1D np.ndarray")

        pw.whisper_pcm_to_mel(self._ctx, media, len(media), n_threads)
        lang_max_id = self.lang_max_id()
        probs = np.zeros(lang_max_id, dtype=np.float32)
        auto_detect = pw.whisper_lang_auto_detect(
            self._ctx, offset_ms, n_threads, probs
        )
        langs = self.available_languages()
        lang_probs = {langs[i]: probs[i] for i in range(lang_max_id)}
        return (langs[auto_detect], probs[auto_detect]), lang_probs

    def __del__(self):
        """
        Free up resources
        :return: None
        """
        pw.whisper_free(self._ctx)
