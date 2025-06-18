/bin/bash
# Script to set up environment and run all required commands sequentially

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pdf_spec_parser/spec_parser.py
python "notebook/sythetic_data gereation_time_series_model.py"
python "notebook/sythetic_data gereation_time_series_model.py"
python notebook/model.py
source venv/bin/activate
python notebook/rul_prod_data.py
chmod +x start.sh
bash start.sh
 #!/bin/bash
# Script to set up environment and run all required commands sequentially

# Initialize a Python virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pdf_spec_parser/spec_parser.py
python "notebook/sythetic_data gereation_time_series_model.py"
python notebook/model.py
source venv/bin/activate
python notebook/rul_prod_data.py
chmod +x start.sh
bash start.sh


# Docker commands to set up and run the application
# docker ps               # to view running containers
docker-compose down -v --remove-orphans
docker container prune -f
docker volume prune -f
docker network prune -f
docker rmi $(docker images | predictive-maintenance-dev | awk '{print $3}')
docker-compose build --no-cache
docker-compose up --force-recreate
