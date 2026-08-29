from __future__ import annotations

from itertools import product
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

NUMBER_OF_EXAMPLES = 250
MAX_STRING_LENGTH = 14
OUTPUT_FILE = Path(__file__).with_name("grammar_training_data_1.xlsx")

# Original grammar:
#   S -> B | C
#   B -> a S b | a b
#   C -> a S c | a c
#
# Equivalent strict Chomsky normal form grammar:
#   S   -> T_a X_b | T_a T_b | T_a X_c | T_a T_c
#   X_b -> S T_b
#   X_c -> S T_c
#   T_a -> a
#   T_b -> b
#   T_c -> c
#
# Every generated string has the form:
#   a^n w, where n >= 1 and w is any length-n string over {b, c}.
#
# Because the converted grammar is in strict CNF, a completed string of
# length m requires exactly 2m - 1 production-rule applications.


def generate_examples() -> list[tuple[str, int]]:
    examples: list[str] = []

    # Strings have even lengths 2n. Generate shorter strings first.
    for n in range(1, (MAX_STRING_LENGTH // 2) + 1):
        for suffix in product("bc", repeat=n):
            value = ("a" * n) + "".join(suffix)
            examples.append(value)

            if len(examples) == NUMBER_OF_EXAMPLES:
                return [
                    (string, 2 * len(string) - 1)
                    for string in examples
                ]

    raise ValueError(
        f"Only {len(examples)} unique strings exist at length "
        f"{MAX_STRING_LENGTH} or less; {NUMBER_OF_EXAMPLES} were requested."
    )


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
                    f'<c r="{reference}" t="inlineStr">'
                    f'<is><t>{text}</t></is></c>'
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
