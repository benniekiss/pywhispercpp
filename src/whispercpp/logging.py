import ctypes
import logging
import os
import sys

from pywhispercpp.lib import _pywhispercpp as pw

logger = logging.getLogger("pywhispercpp")

# Avoid "LookupError: unknown encoding: ascii" when open() called in a destructor
outnull_file = open(os.devnull, "w")
errnull_file = open(os.devnull, "w")

STDOUT_FILENO = 1
STDERR_FILENO = 2


# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_INFO  = 1,
#     GGML_LOG_LEVEL_WARN  = 2,
#     GGML_LOG_LEVEL_ERROR = 3,
#     GGML_LOG_LEVEL_DEBUG = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };
GGML_LOG_LEVEL_TO_LOGGING_LEVEL = {
    0: logging.CRITICAL,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.DEBUG,
    5: logging.DEBUG,
}

_last_log_level = GGML_LOG_LEVEL_TO_LOGGING_LEVEL[0]


def whisper_log_callback(
    level: int,
    text: str,
    user_data: ctypes.c_void_p,
):
    # TODO: Correctly implement continue previous log
    global _last_log_level
    log_level = (
        GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level] if level != 5 else _last_log_level
    )
    if logger.level <= GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level]:
        print(text, end="", flush=True, file=sys.stderr)
    _last_log_level = log_level


pw.assign_whisper_log_callback(whisper_log_callback)


class suppress_stdout_stderr:
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os

    def __init__(self, disable: bool = True):
        self.disable = disable

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        if self.disable:
            return self

        self.old_stdout_fileno_undup = STDOUT_FILENO
        self.old_stderr_fileno_undup = STDERR_FILENO

        self.old_stdout_fileno = self.os.dup(self.old_stdout_fileno_undup)
        self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)

        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.os.dup2(outnull_file.fileno(), self.old_stdout_fileno_undup)
        self.os.dup2(errnull_file.fileno(), self.old_stderr_fileno_undup)

        self.sys.stdout = outnull_file
        self.sys.stderr = errnull_file
        return self

    def __exit__(self, *_):
        if self.disable:
            return

        # Check if sys.stdout and sys.stderr have fileno method
        self.sys.stdout = self.old_stdout
        self.sys.stderr = self.old_stderr

        self.os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        self.os.close(self.old_stdout_fileno)
        self.os.close(self.old_stderr_fileno)
