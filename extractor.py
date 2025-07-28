import fitz
import re

def is_valid_heading(text):
    """Apply strict rules to determine whether a line is a relevant heading."""
    if not text or len(text.split()) > 20 or len(text.split()) < 3:
        return False
    if text.lower() in {"overview", "contents", "acknowledgements", "introduction"}:
        return False
    if text.endswith("."):
        return False
    if re.fullmatch(r"[\d\.\-\/ ]+", text):  # just numbers and dots
        return False
    if re.search(r"\bversion\b|\bpage\b|copyright", text.lower()):
        return False

    # 🚫 Stronger date patterns
    date_patterns = [
        r"\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    ]
    for pat in date_patterns:
        if re.fullmatch(pat, text.strip().lower()):
            return False

    return True

def infer_level(text):
    match = re.match(r"^(\d+(\.\d+){0,2})\s+(.*)", text)
    if match:
        depth = match.group(1).count(".")
        if depth == 0:
            return "H1"
        elif depth == 1:
            return "H2"
        else:
            return "H3"
    return None

def clean_text(text):
    return re.sub(r"^(\d+\.)+\s*", "", text).strip()

def extract_title(page):
    """Extract top 2 lines from first page as title."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = " ".join(span["text"] for span in line["spans"]).strip()
            if text and not re.match(r"^(version|copyright|page)", text.lower()):
                lines.append(text)
            if len(lines) == 2:
                return "  ".join(lines) + "  "
    return "Unknown Title"

def extract_headings(doc):
    seen = set()
    headings = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                text = " ".join(span["text"] for span in line["spans"]).strip()
                if not is_valid_heading(text):
                    continue
                if text in seen:
                    continue
                level = infer_level(text)
                if level:
                    headings.append({
                        "level": level,
                        "text": text.strip() + " ",
                        "page": page_num
                    })
                    seen.add(text)
    return headings

def process_pdf(file_path):
    doc = fitz.open(file_path)
    return {
        "title": extract_title(doc[0]),
        "outline": extract_headings(doc)
    }