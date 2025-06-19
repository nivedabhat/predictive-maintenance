import os
import re
import json
import logging
import pdfplumber
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

UNIT_NORMALIZATION = {
    "KW": "kW", "kw": "kW",
    "RPM": "rpm", "r/min": "rpm",
    "Hz": "Hz",
    "V": "V", "kV": "kV",
    "A": "A",
    "°C": "°C", "C": "°C",
    "s": "s",
    "Nm": "Nm",
    "kg-m2": "kg-m2",
    "dB(A)": "dB(A)",
    "IP55": "IP", "IP56": "IP", "IP66": "IP",
    "kg": "kg"
}


class CleanSpecParser:
    """Parser to extract and clean specification parameters from PDF documents."""

    def __init__(self, pdf_folder: str, output_folder: str):
        """
        Initialize parser with input PDF folder and output folder.

        Args:
            pdf_folder (str): Directory containing PDF files to parse.
            output_folder (str): Directory to save extracted CSV and JSON files.
        """
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder

    def normalize_unit(self, unit: str) -> str:
        """Normalize unit strings using UNIT_NORMALIZATION mapping.

        Args:
            unit (str): Raw unit string.

        Returns:
            str: Normalized unit string or original if no match.
        """
        if not unit:
            return ""
        normalized = unit.strip().replace('¬∞', '°')
        return UNIT_NORMALIZATION.get(normalized, normalized)

    def parse_bounds(self, value: str):
        """Parse numeric bounds from specification value string.

        Handles ±, ranges, inequalities, and exact values with tolerance.

        Args:
            value (str): Value string to parse.

        Returns:
            tuple: (lower bound float or None, upper bound float or None)
        """
        try:
            if "±" in value:
                base, delta = value.split("±")
                lower = float(base.strip()) - float(delta.strip())
                upper = float(base.strip()) + float(delta.strip())
                return lower, upper

            if "-" in value and re.match(r"\d+\.?\d*\s*-\s*\d+\.?\d*", value):
                low, high = map(float, re.split(r"\s*-\s*", value))
                return low, high

            if "≤" in value or "<=" in value:
                upper = float(re.findall(r"\d+\.?\d*", value)[0])
                return None, upper

            if "≥" in value or ">=" in value:
                lower = float(re.findall(r"\d+\.?\d*", value)[0])
                return lower, None

            if re.match(r"^\d+\.?\d*$", value):
                val = float(value)
                return val - 3.0, val + 3.0

        except (ValueError, IndexError) as err:
            logging.debug("Failed to parse bounds for value '%s': %s", value, err)
            return None, None

        return None, None

    def extract_model_id(self, text: str) -> str:
        """Extract model ID from the raw text.

        Args:
            text (str): Full text extracted from PDF.

        Returns:
            str: Extracted model ID or 'UnknownModel'.
        """
        match = re.search(
            r"Product code\s*:?\s*(3GBA[\w\s\-]*)", text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return "UnknownModel"

    def extract_from_text(self, text: str) -> list:
        """Extract parameter rows from raw text.

        Args:
            text (str): Raw extracted text from PDF.

        Returns:
            list: List of dicts with model_id, parameter, value, unit, lower, upper.
        """
        rows = []
        model_id = self.extract_model_id(text)

        for line in text.splitlines():
            line = line.strip()
            if not line or not re.match(r"^\d+\s", line):
                continue

            if len(re.findall(r"\d+\.?\d*", line)) > 5:
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            unit = parts[-1]
            value = parts[-2]
            param = " ".join(parts[1:-2])

            if not re.search(r"[a-zA-Z]", param) or not re.match(r"[\d±≤≥\-\.]+", value):
                continue

            lower, upper = self.parse_bounds(value)

            rows.append({
                "model_id": model_id,
                "parameter": param,
                "value": value,
                "unit": self.normalize_unit(unit),
                "lower": lower,
                "upper": upper
            })

        return rows

    def extract_from_pdf(self, pdf_path: str) -> list:
        """Extract text from PDF and parse parameters.

        Args:
            pdf_path (str): Path to PDF file.

        Returns:
            list: Parsed parameter rows or empty list on error.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return self.extract_from_text(text)

        except (FileNotFoundError, IOError) as err:
            logging.error("Error reading %s: %s", pdf_path, err)
            return []

    def process(self):
        """Process all PDFs in input folder, parse, and save results."""
        all_rows = []

        for file in os.listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, file)
                logging.info("Processing %s", file)
                extracted = self.extract_from_pdf(pdf_path)
                all_rows.extend(extracted)

        logging.info("Total extracted rows: %d", len(all_rows))

        df = pd.DataFrame(all_rows)
        expected_columns = [
            "model_id", "parameter", "value", "unit", "lower", "upper"
        ]

        if all(col in df.columns for col in expected_columns):
            df = df[expected_columns]
        else:
            logging.warning(
                "Missing expected columns in parsed data. Found columns: %s",
                df.columns.tolist()
            )

        csv_path = os.path.join(self.output_folder, "final_clean_parameters.csv")
        json_path = os.path.join(self.output_folder, "final_clean_parameters.json")

        df.to_csv(csv_path, index=False)

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(all_rows, json_file, indent=4)

        logging.info(
            "Saved %s and %s",
            os.path.basename(csv_path),
            os.path.basename(json_path)
        )


if __name__ == "__main__":
    parser = CleanSpecParser(
        pdf_folder="pdf_spec_parser/sample_pdfs",
        output_folder="pdf_spec_parser/output"
    )
    parser.process()

