"""GCS object transfer using the managed Vertex runtime's metadata credentials."""

from pathlib import Path
from urllib.parse import quote, urlparse

import requests


def object_parts(uri: str) -> tuple[str, str]:
    value = urlparse(uri)
    if value.scheme != "gs" or not value.netloc or not value.path.lstrip("/"):
        raise ValueError("Expected gs://bucket/object")
    return value.netloc, value.path.lstrip("/")


def headers() -> dict[str, str]:
    response = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
        timeout=15,
    )
    response.raise_for_status()
    return {"Authorization": "Bearer " + response.json()["access_token"]}


def download(uri: str, destination: Path) -> None:
    bucket, name = object_parts(uri)
    url = f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o/{quote(name, safe='')}"
    with requests.get(url, params={"alt": "media"}, headers=headers(), stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)


def upload(source: Path, uri: str) -> None:
    bucket, name = object_parts(uri)
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o"
    with source.open("rb") as data:
        response = requests.post(
            url,
            params={"uploadType": "media", "name": name},
            headers={**headers(), "Content-Type": "application/octet-stream"},
            data=data,
            timeout=600,
        )
    response.raise_for_status()
