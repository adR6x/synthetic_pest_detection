import re
import requests
from bs4 import BeautifulSoup


def print_secret_message(url: str) -> None:
    """
    Fetches a published Google Doc, parses the (x, y, character) table,
    and prints the resulting character grid to reveal a hidden message.

    Args:
        url: URL of the Google Doc — works with both regular edit URLs
             and published /pub URLs.
    """
    # Published /pub URLs can be fetched directly as HTML.
    # Regular /document/d/{id}/... URLs are converted to export?format=html.
    if url.rstrip("/").endswith("/pub") or "/pub?" in url:
        fetch_url = url
    else:
        match = re.search(r'/document/d/(?!e/)([a-zA-Z0-9_-]+)', url)
        if not match:
            raise ValueError(f"Cannot extract document ID from URL: {url}")
        doc_id = match.group(1)
        fetch_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"

    response = requests.get(fetch_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    data = []   # list of (x, y, char)

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        # Detect column positions from the header row
        header_cells = rows[0].find_all(["th", "td"])
        headers = [c.get_text(strip=True).lower() for c in header_cells]

        x_col = char_col = y_col = None
        for i, h in enumerate(headers):
            if "x" in h and "coord" in h:
                x_col = i
            elif "y" in h and "coord" in h:
                y_col = i
            elif "char" in h:
                char_col = i

        # Fall back to documented column order if detection fails
        if None in (x_col, char_col, y_col):
            x_col, char_col, y_col = 0, 1, 2

        for row in rows[1:]:
            cells = row.find_all("td")
            if len(cells) <= max(x_col, char_col, y_col):
                continue
            try:
                x    = int(cells[x_col].get_text(strip=True))
                y    = int(cells[y_col].get_text(strip=True))
                char = cells[char_col].get_text(strip=True)
                if char:
                    data.append((x, y, char))
            except (ValueError, IndexError):
                continue

    if not data:
        print("No character data found in the document.")
        return

    max_x = max(item[0] for item in data)
    max_y = max(item[1] for item in data)

    # grid[y][x]; y=0 is top row, x=0 is leftmost column
    grid = [[" "] * (max_x + 1) for _ in range(max_y + 1)]
    for x, y, char in data:
        grid[y][x] = char

    for row in grid:
        print("".join(row))


if __name__ == "__main__":
    print_secret_message(
        "https://docs.google.com/document/d/e/"
        "2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"
    )
