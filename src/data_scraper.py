"""GBIF image scraper for aquatic species classification."""

from __future__ import annotations

import os
from pathlib import Path

import requests


GBIF_API_URL = "https://api.gbif.org/v1/occurrence/search"
DEFAULT_MAX_IMAGES = 500
GBIF_PAGE_LIMIT = 300


def fetch_image_urls(species_name: str) -> list[str]:
    params: dict[str, int | str] = {
        "scientificName": species_name,
        "mediaType": "StillImage",
        "limit": GBIF_PAGE_LIMIT,
        "offset": 0,
    }
    urls: list[str] = []

    while True:
        response = requests.get(GBIF_API_URL, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            break
        for record in results:
            for media in record.get("media", []):
                if media.get("type") == "StillImage" and (url := media.get("identifier")):
                    urls.append(url)
        params["offset"] = int(params["offset"]) + GBIF_PAGE_LIMIT

    return urls


def download_images(
    species_name: str,
    output_dir: Path,
    max_images: int = DEFAULT_MAX_IMAGES,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    urls = fetch_image_urls(species_name)
    downloaded = 0

    for idx, url in enumerate(urls):
        if downloaded >= max_images:
            break
        try:
            img_data = requests.get(url, timeout=10).content
            (output_dir / f"img_{idx:04d}.jpg").write_bytes(img_data)
            downloaded += 1
        except requests.RequestException:
            continue

    print(f"Downloaded {downloaded} images for {species_name}")
    return downloaded


if __name__ == "__main__":
    base = Path("dataset")
    download_images("Amphiprion ocellaris", base / "target", max_images=500)
    download_images("Amphiprion clarkii", base / "non_target", max_images=300)
    download_images("Neoglyphidodon oxyodon", base / "non_target", max_images=300)
