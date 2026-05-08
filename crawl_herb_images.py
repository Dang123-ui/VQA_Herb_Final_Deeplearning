"""
Script to crawl herb images from the website tracuuduoclieu.vn

The script iterates through the listing pages under the path
`/tra-cuu-duoc-lieu`, follows each herb's detail link and
collects all images found inside the `#detail-dl` element.  Every
image is downloaded locally and metadata about the image is
written to a JSON file.  The JSON objects include an `id`
generated from a running counter (zero‑padded to nine digits and
prefixed with ``P``) and the plant's Vietnamese name.  The first
image will have the identifier ``P000000001``.

To run the script, ensure that you have the required Python
packages installed (``requests`` and ``beautifulsoup4``).  You can
install them via ``pip install -r requirements.txt`` or manually.

Example usage::

    python crawl_herb_images.py

This will create a directory called ``herb_images`` in the same
folder as the script (if it doesn’t already exist) and store the
downloaded pictures there.  A JSON file named
``herb_images.json`` will be created to record each image’s
identifier and corresponding herb name.
"""

import json
import os
import re
import sys
from typing import Dict, List, Set

import requests
from bs4 import BeautifulSoup


# Base URL for the site
BASE_URL = "https://tracuuduoclieu.vn"
LISTING_PATH = "/tra-cuu-duoc-lieu"

# HTTP headers to mimic a typical browser.  Some websites block
# unknown or empty User‑Agents; using a common browser string
# helps avoid simple blocking.
HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "vi,vi-VN;q=0.9,en;q=0.8",
    "Referer": BASE_URL,
}


def get_soup(url: str) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object.

    Args:
        url: The page URL to fetch.

    Returns:
        A BeautifulSoup instance representing the fetched HTML.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching {url}: {exc}", file=sys.stderr)
        raise
    return BeautifulSoup(resp.text, "html.parser")


def extract_herb_links(listing_soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Extract herb names and detail links from a listing page.

    Args:
        listing_soup: Soup of the herb listing page.

    Returns:
        A list of dictionaries with keys ``name`` and ``url``.
    """
    herbs: List[Dict[str, str]] = []

    # On listing pages each herb entry resides inside a div with the class
    # "dl".  The structure is roughly:
    #   <div class="dl">
    #       <a href="..."><img ...></a>
    #       <h2><a href="...">Herb Name</a></h2>
    #   </div>
    for dl_div in listing_soup.select("div.dl"):
        h2 = dl_div.find("h2")
        link = h2.find("a") if h2 else None
        if not link:
            continue
        name = link.get_text(strip=True)
        url = link.get("href")
        if not url:
            continue
        # Ensure absolute URLs
        if url.startswith("/"):
            url = BASE_URL + url
        herbs.append({"name": name, "url": url})
    return herbs


def extract_images_from_detail(detail_soup: BeautifulSoup) -> List[str]:
    """Extract image URLs from a herb detail page.

    Only images within the `#detail-dl` element are returned.  Where
    both ``data-src`` and ``src`` attributes are present, ``data-src`` is
    preferred (because it often holds the original image whereas
    ``src`` may contain a down‑sampled thumbnail).

    Args:
        detail_soup: Soup of the herb detail page.

    Returns:
        A list of absolute image URLs.
    """
    image_urls: List[str] = []
    detail_container = detail_soup.find(id="detail-dl")
    if not detail_container:
        return image_urls
    for img in detail_container.find_all("img"):
        # Some images are lazy‑loaded via data‑src; fall back to src if necessary.
        img_url = img.get("data-src") or img.get("src")
        if not img_url:
            continue
        # Normalize and build absolute URLs
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            img_url = BASE_URL + img_url
        # Remove query strings for clean filenames
        img_url = img_url.split("?")[0]
        image_urls.append(img_url)
    return image_urls


def download_image(img_url: str, dest_folder: str, filename: str) -> None:
    """Download a single image to the destination folder.

    Args:
        img_url: The absolute URL of the image to download.
        dest_folder: Path to the directory where the image should be saved.
        filename: The filename to use for the saved image.
    """
    try:
        resp = requests.get(img_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed to download {img_url}: {exc}", file=sys.stderr)
        return
    out_path = os.path.join(dest_folder, filename)
    with open(out_path, "wb") as f:
        f.write(resp.content)


def main() -> None:
    # Output paths
    images_dir = os.path.join(os.getcwd(), "herb_images")
    os.makedirs(images_dir, exist_ok=True)
    json_path = os.path.join(os.getcwd(), "herb_images.json")

    # Prepare collection data and helper structures
    collected: List[Dict[str, str]] = []
    downloaded_urls: Set[str] = set()
    id_counter = 1

    # Determine the number of pages to crawl.  The site currently
    # advertises up to 61 pages.  This value can be adjusted if the
    # site adds or removes pages over time.
    max_pages = 30

    for page in range(1, max_pages + 1):
        if page == 1:
            page_url = BASE_URL + LISTING_PATH
        else:
            page_url = f"{BASE_URL}{LISTING_PATH}/page/{page}"
        try:
            listing_soup = get_soup(page_url)
        except Exception:
            # Skip this page if it fails to load
            print(f"Skipping page {page_url} due to fetch error.", file=sys.stderr)
            continue
        herbs = extract_herb_links(listing_soup)
        if not herbs:
            # If no herbs were found on the current page, it may indicate
            # the end of the listing.  Break early to avoid unnecessary
            # requests.
            break
        for herb in herbs:
            herb_name = herb["name"]
            herb_url = herb["url"]
            try:
                detail_soup = get_soup(herb_url)
            except Exception:
                print(f"Skipping detail {herb_url} due to fetch error.", file=sys.stderr)
                continue
            img_urls = extract_images_from_detail(detail_soup)
            for img_url in img_urls:
                # Avoid downloading the same image more than once
                if img_url in downloaded_urls:
                    continue
                downloaded_urls.add(img_url)
                # Determine the file extension
                ext_match = re.search(r"\.([a-zA-Z0-9]+)$", img_url)
                ext = ext_match.group(0) if ext_match else ".jpg"
                # Generate ID and filename
                image_id = f"P{id_counter:09d}"
                filename = image_id + ext
                download_image(img_url, images_dir, filename)
                # Record metadata
                collected.append({"id": image_id, "name": herb_name})
                id_counter += 1

    # Write JSON output
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(collected, jf, ensure_ascii=False, indent=2)
    print(f"Crawling finished. Saved {len(collected)} images and metadata to {json_path}.")


if __name__ == "__main__":
    main()