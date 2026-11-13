"""The installed stack is the pinned one, and the interpreter can load saved models."""
import io
import sys
import zipfile

import gymnasium
import sb3_contrib
import stable_baselines3


def test_pinned_versions():
    assert sys.version_info >= (3, 11)
    assert gymnasium.__version__ == "1.4.0"
    assert stable_baselines3.__version__ == "2.9.2"
    assert sb3_contrib.__version__ == "2.9.2"


def test_zipfile_streams_seek_correctly():
    """SB3 loads models by letting torch seek inside a zip member. Python 3.12.0's zipfile returns
    no data (or raises EOFError) for a seek after the end of the member was reached, which makes
    every SB3 `.load()` fail with "PytorchStreamReader failed reading file". If this test fails,
    the interpreter must be replaced by a later patch release; it is not a bug in this project."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.bin", bytes(range(256)) * 8)
    z = zipfile.ZipFile(buf)
    raw, f = z.read("a.bin"), z.open("a.bin")
    try:
        f.seek(2000); f.read()
        f.seek(1000); first = f.read(30)
        f.seek(1500); second = f.read(40)
    except EOFError:
        first = second = b""
    assert first == raw[1000:1030] and second == raw[1500:1540], (
        f"zipfile stream bug in Python {sys.version.split()[0]}: SB3 models cannot be loaded")
