import os
import re
import json
import logging
import pdfplumber
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    def __init__(self, pdf_folder: str, output_folder: str):
        self.pdf_folder = pdf_folder
        self.output_folder = output_folder

    def normalize_unit(self, unit: str) -> str:
        if not unit:
            return ""
        return UNIT_NORMALIZATION.get(unit.strip().replace('¬∞', '°'), unit.strip().replace('¬∞', '°'))

    def parse_bounds(self, value: str):
        try:
            if "±" in value:
                base, delta = value.split("±")
                return float(base.strip()) - float(delta.strip()), float(base.strip()) + float(delta.strip())
            elif "-" in value and re.match(r"\d+\.?\d*\s*-\s*\d+\.?\d*", value):
                low, high = map(float, re.split(r"\s*-\s*", value))
                return low, high
            elif "≤" in value or "<=" in value:
                upper = float(re.findall(r"\d+\.?\d*", value)[0])
                return None, upper
            elif "≥" in value or ">=" in value:
                lower = float(re.findall(r"\d+\.?\d*", value)[0])
                return lower, None
            elif re.match(r"^\d+\.?\d*$", value):
                val = float(value)
                return val - 3.0, val + 3.0
        except:
            return None, None
        return None, None

    def extract_model_id(self, text: str) -> str:
        match = re.search(r"Product code\s*:?\s*(3GBA[\w\s\-]*)", text, re.IGNORECASE)
        return match.group(1).strip() if match else "UnknownModel"

    def extract_from_text(self, text: str) -> list:
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
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            return self.extract_from_text(text)
        except Exception as e:
            logging.error(f"Error reading {pdf_path}: {e}")
            return []

    def process(self):
        all_rows = []
        for file in os.listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(self.pdf_folder, file)
                logging.info(f"Processing {file}")
                extracted = self.extract_from_pdf(pdf_path)
                all_rows.extend(extracted)

        logging.info(f"Total extracted rows: {len(all_rows)}")
        df = pd.DataFrame(all_rows)
        expected_columns = ["model_id", "parameter", "value", "unit", "lower", "upper"]
        if all(col in df.columns for col in expected_columns):
            df = df[expected_columns]
        else:
            logging.warning(f"Missing expected columns in parsed data. Found columns: {df.columns.tolist()}")
        df.to_csv(os.path.join(self.output_folder, "final_clean_parameters.csv"), index=False)
        with open(os.path.join(self.output_folder, "final_clean_parameters.json"), "w") as f:
            json.dump(all_rows, f, indent=4)
        logging.info("Saved final_clean_parameters.csv and final_clean_parameters.json")

if __name__ == "__main__":
    parser = CleanSpecParser(
        pdf_folder="/Users/niveda/Desktop/PredictiveMaintenanceProject/pdf_spec_parser/sample_pdfs",
        output_folder="/Users/niveda/Desktop/PredictiveMaintenanceProject/pdf_spec_parser/output"
    )
    parser.process()

