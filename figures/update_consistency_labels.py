"""Repair labels in the existing vector PDFs without redrawing their contents.

Run from the repository root with pypdf installed. The PDFs themselves are the
editable vector source; already corrected files are left unchanged.
"""

from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ContentStream,
    FloatObject,
    NameObject,
    NumberObject,
    TextStringObject,
)


ROOT = Path(__file__).resolve().parents[1]


def text_block(text, x, y, size):
    return [
        ([], b"BT"),
        ([NameObject("/F2"), FloatObject(size)], b"Tf"),
        ([FloatObject(v) for v in (1, 0, 0, 1, x, y)], b"Tm"),
        ([FloatObject(0)], b"g"),
        ([FloatObject(0)], b"G"),
        ([TextStringObject(text)], b"Tj"),
        ([], b"ET"),
    ]


def update_pdf(name, regions):
    path = ROOT / "els-cas-templates" / name
    reader = PdfReader(path)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    page = writer.pages[0]
    extracted = page.extract_text()
    font = page["/Resources"]["/Font"]["/F2"].get_object()
    widths = font["/Widths"].get_object()
    first = int(font["/FirstChar"])
    # PowerPoint leaves unused glyph widths zero or beyond /LastChar. These
    # Times New Roman metrics match the original PDF's rounded 1000-em units.
    extra_widths = {"F": 556, "T": 611, "z": 444}
    metrics_changed = False
    for character in {c for _, lines in regions for line in lines for c in line[0]}:
        index = ord(character) - first
        while index >= len(widths):
            widths.append(NumberObject(0))
        if not widths[index]:
            widths[index] = NumberObject(extra_widths[character])
            metrics_changed = True
    font[NameObject("/LastChar")] = NumberObject(first + len(widths) - 1)
    if not metrics_changed and all(line[0] in extracted for _, lines in regions for line in lines):
        return
    operations = page.get_contents().operations
    replacements = {}
    for (xmin, ymin, xmax, ymax), lines in regions:
        selected = []
        start = None
        for i, (args, op) in enumerate(operations):
            if op == b"BT":
                start = i
            elif op == b"ET" and start is not None:
                matrices = [a for a, o in operations[start:i] if o == b"Tm"]
                if matrices:
                    x, y = map(float, matrices[0][4:6])
                    if xmin < x < xmax and ymin < y < ymax:
                        selected.append((start, i))
                start = None
        if not selected:
            raise ValueError(f"No existing label found in {name}: {xmin, ymin, xmax, ymax}")
        new = []
        for text, x, y, size, centered in lines:
            if centered:
                width = sum(float(widths[ord(c) - first]) for c in text) * size / 1000
                x -= width / 2
            new.extend(text_block(text, x, y, size))
        for j, (start, end) in enumerate(selected):
            replacements[start] = (end, new if j == 0 else [])
    corrected = []
    i = 0
    while i < len(operations):
        if i in replacements:
            end, new = replacements[i]
            corrected.extend(new)
            i = end + 1
        else:
            corrected.append(operations[i])
            i += 1
    content = ContentStream(None, writer)
    content.operations = corrected
    page.replace_contents(content)
    writer.add_metadata({"/ModDate": datetime.now(timezone.utc).strftime("D:%Y%m%d%H%M%SZ")})
    with path.open("wb") as stream:
        writer.write(stream)
    result = PdfReader(path).pages[0].extract_text()
    for _, lines in regions:
        for line in lines:
            if line[0] not in result:
                raise ValueError(f"Corrected label was not readable: {line[0]}")


if __name__ == "__main__":
    update_pdf("motivation.pdf", [
        ((630, 350, 795, 380), [
            ("Favor similar candidates with", 704.3, 373.8, 11.04, True),
            ("lower train-set skew.", 704.3, 360.6, 11.064, True),
        ]),
        ((760, 80, 940, 95), [
            ("Target utility retention", 765.05, 86.544, 11.04, False),
        ]),
        ((760, 50, 940, 70), [
            ("Assess user-group utility gaps", 766.75, 57.336, 11.04, False),
        ]),
        ((370, 300, 550, 325), [
            ("Post-hoc or in-training score", 374.71, 316.25, 11.04, False),
            ("adjustment.", 374.71, 303.05, 11.04, False),
        ]),
    ])
    update_pdf("Fair-NCL.pdf", [
        ((590, 140, 700, 162), [
            ("Penalize low", 641.44, 158.59, 9.0, True),
            ("per-dimension variance", 641.44, 147.79, 9.024, True),
        ]),
    ])
