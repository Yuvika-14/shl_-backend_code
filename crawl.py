import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin
import time

BASE = "https://www.shl.com"
CATALOG = BASE + "/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch(url):
    """Fetch and parse a webpage."""
    try:
        r = requests.get(url, timeout=15, headers=HEADERS)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None


def extract_assessment_data(url):
    """Extract assessment details from an individual assessment page."""
    soup = fetch(url)
    if not soup:
        return None

    # --- Assessment name ---
    name_tag = soup.find("h1")
    assessment_name = name_tag.get_text(strip=True) if name_tag else None

    # --- Description ---
    desc_tag = soup.find("meta", {"name": "description"})
    description = desc_tag["content"].strip() if desc_tag and "content" in desc_tag.attrs else ""

    # fallback to first paragraph if meta empty
    if not description:
        p = soup.find("p")
        description = p.get_text(strip=True) if p else ""

    # --- Test Type (inferred from URL) ---
    if "/personality-assessment/" in url:
        test_type = "Personality & Behavior"
    elif "/skills-and-simulations/" in url:
        test_type = "Knowledge & Skills"
    elif "/behavioral-assessments/" in url:
        test_type = "Behavioral"
    elif "/cognitive-assessments/" in url:
        test_type = "Cognitive"
    else:
        test_type = "Other"

    # --- Skills / topics (best-effort extraction) ---
    skills = []
    for li in soup.select("ul li"):
        text = li.get_text(strip=True)
        if 2 < len(text) < 60:  # avoid long paragraphs
            skills.append(text)
    skills = list(set(skills))[:8]

    return {
        "assessment_name": assessment_name,
        "url": url,
        "description": description,
        "test_type": test_type,
        "skills": skills,
    }


def crawl():
    print("🔍 Fetching catalog page...")
    soup = fetch(CATALOG)
    if not soup:
        print("❌ Failed to fetch catalog page.")
        return

    # Collect links
    items = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        url = urljoin(BASE, href)
        if "/products/assessments/" not in url:
            continue
        items.append(url)

    items = sorted(list(set(items)))
    print(f"🧩 Found {len(items)} potential assessment URLs")

    # Extract details for each page
    all_data = []
    for i, url in enumerate(items, 1):
        print(f"➡️ [{i}/{len(items)}] {url}")
        data = extract_assessment_data(url)
        if data and data["assessment_name"]:
            all_data.append(data)
        time.sleep(0.5)  # be polite

    # Save to CSV
    df = pd.DataFrame(all_data)
    df.to_csv("catalog_enriched_1.csv", index=False)
    print(f"\n✅ Saved catalog_enriched_1.csv with {len(df)} assessments")


if __name__ == "__main__":
    crawl()
