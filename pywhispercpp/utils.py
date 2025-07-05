import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def to_timestamp(t: int, separator=",") -> str:
    """
    376 -> 00:00:03,760
    1344 -> 00:00:13,440

    Implementation from `whisper.cpp/examples/main`

    :param t: input time from whisper timestamps
    :param separator: seprator between seconds and milliseconds
    :return: time representation in hh: mm: ss[separator]ms
    """
    # logic exactly from whisper.cpp

    msec = t * 10
    hr = msec // (1000 * 60 * 60)
    msec = msec - hr * (1000 * 60 * 60)
    min = msec // (1000 * 60)
    msec = msec - min * (1000 * 60)
    sec = msec // 1000
    msec = msec - sec * 1000
    return f"{int(hr):02,.0f}:{int(min):02,.0f}:{int(sec):02,.0f}{separator}{int(msec):03,.0f}"


def output_txt(segments: list, output_file_path: str) -> str:
    """
    Creates a raw text from a list of segments

    Implementation from `whisper.cpp/examples/main`

    :param segments: list of segments
    :return: path of the file
    """
    if not output_file_path.endswith(".txt"):
        output_file_path = output_file_path + ".txt"

    absolute_path = Path(output_file_path).absolute()

    with open(str(absolute_path), "w") as file:
        for seg in segments:
            file.write(seg.text)
            file.write("\n")
    return absolute_path


def output_vtt(segments: list, output_file_path: str) -> str:
    """
    Creates a vtt file from a list of segments

    Implementation from `whisper.cpp/examples/main`

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith(".vtt"):
        output_file_path = output_file_path + ".vtt"

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, "w") as file:
        file.write("WEBVTT\n\n")
        for seg in segments:
            file.write(
                f"{to_timestamp(seg.t0, separator='.')} --> {to_timestamp(seg.t1, separator='.')}\n"
            )
            file.write(f"{seg.text}\n\n")
    return absolute_path


def output_srt(segments: list, output_file_path: str) -> str:
    """
    Creates a srt file from a list of segments

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith(".srt"):
        output_file_path = output_file_path + ".srt"

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, "w") as file:
        for i in range(len(segments)):
            seg = segments[i]
            file.write(f"{i + 1}\n")
            file.write(
                f"{to_timestamp(seg.t0, separator=',')} --> {to_timestamp(seg.t1, separator=',')}\n"
            )
            file.write(f"{seg.text}\n\n")
    return absolute_path


def output_csv(segments: list, output_file_path: str) -> str:
    """
    Creates a srt file from a list of segments

    :param segments: list of segments
    :return: path of the file

    :return: Absolute path of the file
    """
    if not output_file_path.endswith(".csv"):
        output_file_path = output_file_path + ".csv"

    absolute_path = Path(output_file_path).absolute()

    with open(absolute_path, "w") as file:
        for seg in segments:
            file.write(f'{10 * seg.t0}, {10 * seg.t1}, "{seg.text}"\n')
    return absolute_path
