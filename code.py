import re
from html.parser import HTMLParser
from urllib.error import URLError, HTTPError
from urllib.request import urlopen


class _TableCellParser(HTMLParser):
    """Extracts table cells row-by-row from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._in_tr = False
        self._in_cell = False
        self._current_cell_parts = []
        self._current_row = []
        self.rows = []

    def handle_starttag(self, tag, attrs):  # noqa: D401
        t = tag.lower()
        if t == "tr":
            self._in_tr = True
            self._current_row = []
        elif self._in_tr and t in {"td", "th"}:
            self._in_cell = True
            self._current_cell_parts = []

    def handle_endtag(self, tag):  # noqa: D401
        t = tag.lower()
        if t in {"td", "th"} and self._in_cell:
            text = "".join(self._current_cell_parts).replace("\xa0", " ")
            text = re.sub(r"\s+", " ", text).strip()
            self._current_row.append(text)
            self._in_cell = False
            self._current_cell_parts = []
        elif t == "tr" and self._in_tr:
            if self._current_row:
                self.rows.append(self._current_row)
            self._in_tr = False
            self._current_row = []

    def handle_data(self, data):  # noqa: D401
        if self._in_cell:
            self._current_cell_parts.append(data)


def _download_text(url: str) -> str:
    try:
        with urlopen(url, timeout=30) as resp:
            encoding = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(encoding, errors="replace")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to fetch URL: {url}") from exc


def _build_candidate_urls(url: str) -> list[str]:
    candidates = [url]

    # Edit/view URL (not /d/e/ published variant)
    match_doc = re.search(r"docs\.google\.com/document/d/(?!e/)([A-Za-z0-9_-]+)", url)
    if match_doc:
        doc_id = match_doc.group(1)
        candidates.extend(
            [
                f"https://docs.google.com/document/d/{doc_id}/export?format=html",
                f"https://docs.google.com/document/d/{doc_id}/pub",
            ]
        )

    # Published /d/e/... URL
    match_pub = re.search(r"docs\.google\.com/document/d/e/([A-Za-z0-9_-]+)", url)
    if match_pub:
        pub_id = match_pub.group(1)
        candidates.extend(
            [
                f"https://docs.google.com/document/d/e/{pub_id}/pub",
                f"https://docs.google.com/document/d/e/{pub_id}/pub?embedded=true",
            ]
        )

    # Deduplicate, preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _extract_points_from_rows(rows: list[list[str]]) -> dict[tuple[int, int], str]:
    points: dict[tuple[int, int], str] = {}

    # Find header row and detect indexes dynamically
    for i in range(len(rows)):
        hdr = [c.lower() for c in rows[i]]
        if len(hdr) < 3:
            continue

        x_idx = y_idx = ch_idx = None
        for j, cell in enumerate(hdr):
            if "x" in cell and "coord" in cell:
                x_idx = j
            elif "y" in cell and "coord" in cell:
                y_idx = j
            elif "char" in cell:
                ch_idx = j

        if None in (x_idx, y_idx, ch_idx):
            continue

        # Parse rows after the detected header
        for row in rows[i + 1 :]:
            if max(x_idx, y_idx, ch_idx) >= len(row):
                continue
            xs = row[x_idx]
            ys = row[y_idx]
            ch = row[ch_idx]
            if re.fullmatch(r"\d+", xs) and re.fullmatch(r"\d+", ys):
                points[(int(xs), int(ys))] = ch if ch else " "
        break

    return points


def print_secret_grid(doc_url: str) -> None:
    """
    Takes a Google Doc URL containing (x-coordinate, character, y-coordinate)
    rows and prints the reconstructed character grid.
    """
    html = ""
    last_err = None

    for candidate in _build_candidate_urls(doc_url):
        try:
            html = _download_text(candidate)
            if "x-coordinate" in html.lower() and "y-coordinate" in html.lower():
                break
        except RuntimeError as exc:
            last_err = exc
    else:
        if last_err is not None:
            raise last_err
        raise RuntimeError("Unable to fetch a valid Google Doc page.")

    parser = _TableCellParser()
    parser.feed(html)
    points = _extract_points_from_rows(parser.rows)

    if not points:
        raise ValueError("No coordinate/character rows were found in the document.")

    max_x = max(x for x, _ in points.keys())
    max_y = max(y for _, y in points.keys())

    # Based on the provided example orientation, print from highest y to lowest y.
    for y in range(max_y, -1, -1):
        row = "".join(points.get((x, y), " ") for x in range(max_x + 1))
        print(row)
