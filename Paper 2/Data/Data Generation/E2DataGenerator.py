from __future__ import annotations

from itertools import product
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

NUMBER_OF_EXAMPLES = 250
MAX_STRING_LENGTH = 30
OUTPUT_FILE = Path(__file__).with_name("grammar_training_data_3.xlsx")

# Grammar implemented:
# S   -> A B | T_a A | T_b X | T_b T_c | a
# A   -> T_a A | a
# B   -> T_b X | T_b T_c
# X   -> B T_c
# T_a -> a
# T_b -> b
# T_c -> c
#
# Its language is:
#   a^p                    for p >= 1
#   b^n c^n                for n >= 1
#   a^p b^n c^n            for p,n >= 1
#
# The grammar is in Chomsky normal form. A completed string of length m
# therefore requires exactly 2m - 1 production-rule applications.


def generate_examples() -> list[tuple[str, int]]:
    examples: set[str] = set()

    for length in range(1, MAX_STRING_LENGTH + 1):
        # S => a or S => T_a A, producing a^length.
        examples.add("a" * length)

        # S derives b^n c^n when the total length is even.
        if length % 2 == 0:
            n = length // 2
            examples.add(("b" * n) + ("c" * n))

        # S => A B, producing a^p b^n c^n.
        for n in range(1, (length - 1) // 2 + 1):
            p = length - (2 * n)
            if p >= 1:
                examples.add(("a" * p) + ("b" * n) + ("c" * n))

    ordered = sorted(examples, key=lambda value: (len(value), value))
    if len(ordered) < NUMBER_OF_EXAMPLES:
        raise ValueError(
            f"Only {len(ordered)} unique strings exist at length "
            f"{MAX_STRING_LENGTH} or less; {NUMBER_OF_EXAMPLES} were requested."
        )

    selected = ordered[:NUMBER_OF_EXAMPLES]
    return [(value, 2 * len(value) - 1) for value in selected]


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def make_sheet_xml(rows: list[list[object]]) -> str:
    xml_rows: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_number, value in enumerate(row, start=1):
            reference = f"{excel_column_name(column_number)}{row_number}"
            if isinstance(value, int):
                cells.append(f'<c r="{reference}"><v>{value}</v></c>')
            else:
                text = escape(str(value))
                cells.append(
                    f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        '</worksheet>'
    )


def write_plain_xlsx(path: Path, examples: list[tuple[str, int]]) -> None:
    rows: list[list[object]] = [
        ["Generated String", "Production Rule Applications"],
        *[[value, applications] for value, applications in examples],
    ]

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''

    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Training Data" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''

    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''

    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", make_sheet_xml(rows))


def main() -> None:
    examples = generate_examples()
    write_plain_xlsx(OUTPUT_FILE, examples)

    print(f"Generated {len(examples)} unique examples.")
    print(f"Longest included string: {max(len(value) for value, _ in examples)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
