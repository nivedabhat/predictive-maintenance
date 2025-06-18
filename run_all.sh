python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pdf_spec_parser/spec_parser.py
python "notebook/sythetic_data gereation_time_series_model.py"
python notebook/model.py
source venv/bin/activate
python notebook/rul_prod_data.py
