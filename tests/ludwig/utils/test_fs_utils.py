import logging
import os
import platform
import tempfile
from urllib.parse import quote

import fsspec
import pytest
import s3fs

from ludwig.constants import CLIENT_KWARGS, KEY, SECRET
from ludwig.utils.fs_utils import get_fs_and_path


def create_file(url):
    _, path = get_fs_and_path(url)
    logging.info(f"saving url '{url}' to path '{path}'")
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, path)
        os.makedirs(os.path.dirname(file_path))
        with open(file_path, "w"):
            return path


@pytest.mark.filesystem
def test_get_fs_and_path_simple():
    assert create_file("http://a/b.jpg") == os.path.join("a", "b.jpg")


@pytest.mark.filesystem
def test_get_fs_and_path_query_string():
    assert create_file("http://a/b.jpg?c=d") == os.path.join("a", "b.jpg")


@pytest.mark.filesystem
def test_get_fs_and_path_decode():
    assert create_file("http://a//b%20c.jpg") == os.path.join("a", "b c.jpg")


@pytest.mark.filesystem
def test_get_fs_and_path_unicode():
    assert create_file("http://a/æ.jpg") == "a/æ.jpg"


@pytest.mark.filesystem
def test_get_fs_and_path_with_local_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        fs, path = get_fs_and_path(tmpdir)
        assert isinstance(fs, fsspec.implementations.local.LocalFileSystem)
        assert path == tmpdir


@pytest.mark.filesystem
@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping if windows.")
def test_get_fs_and_path_invalid_linux():
    invalid_chars = {
        "\x00": ValueError,
        "/": FileExistsError,
    }
    for c, e in invalid_chars.items():
        url = f"http://a/{quote(c)}"
        with pytest.raises(e):
            create_file(url)


@pytest.mark.filesystem
@pytest.mark.skipif(platform.system() != "Windows", reason="Skipping if not windows.")
def test_get_fs_and_path_invalid_windows():
    invalid_chars = {
        "\x00": ValueError,
        "\\": FileExistsError,
        "/": OSError,
        ":": OSError,
        "*": OSError,
        "?": OSError,
        '"': OSError,
        "<": OSError,
        ">": OSError,
        "|": OSError,
    }
    for c, e in invalid_chars.items():
        url = f"http://a/{quote(c)}"
        with pytest.raises(e):
            create_file(url)


@pytest.mark.filesystem
def test_get_fs_and_path_with_storage_options():
    bucket_name = "invalid-bucket"

    # No storage options
    fs, path = get_fs_and_path(f"s3://{bucket_name}")
    assert isinstance(fs, s3fs.core.S3FileSystem)
    assert path == bucket_name

    # Empty storage options
    storage_options = None
    fs, path = get_fs_and_path(f"s3://{bucket_name}", storage_options=storage_options)
    assert isinstance(fs, s3fs.core.S3FileSystem)
    assert path == bucket_name

    # Empty storage options
    storage_options = {KEY: "", SECRET: "", CLIENT_KWARGS: ""}
    fs, path = get_fs_and_path(f"s3://{bucket_name}", storage_options=storage_options)
    assert isinstance(fs, s3fs.core.S3FileSystem)
    assert path == bucket_name

    # Fake credentials - should still initialize correctly
    storage_options = {SECRET: "456", CLIENT_KWARGS: "789"}
    fs, path = get_fs_and_path(f"s3://{bucket_name}", storage_options=storage_options)
    assert isinstance(fs, s3fs.core.S3FileSystem)
    assert path == bucket_name
