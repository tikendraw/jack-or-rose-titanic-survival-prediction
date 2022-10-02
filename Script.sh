#! bin/bash

#
echo 'Creating Env'
python3 -m venv titanic_env

echo 'Activating Env'
source titanic_env/bin/activate

echo 'Install repository'
pip install -r requirements.txt

echo 'Running the App'
streamlit run titanicapp.py
