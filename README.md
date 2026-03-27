# twintumor
TwinTumor – Digital Twins for Tumor Growth and Treatment Simulation

Run the rule-based longitude single-case pipeline

1. Create virtual environment
MacOS/Linux: python3 -m venv .venv
Windows: python -m venv .venv

2. Activate the virtual environment
MacOS/Linux: source .venv/bin/activate
Windows(Command Promt): .venv\Scripts\activate
Windows(PowerShell): .venv\Scripts\Activate.ps1

After activation you should see (.venv) at the beginning of the terminal line.

3. Install required packages
pip install numpy pandas nibabel

Open src/pipelines/longitudinal_run.py and update:
DATASET_ROOT = Path("/path/to/your/series")

DATASET_ROOT must point to the folder that contains:
Mets_XXX subject folders
tumor_volumes_all_subjects_v3.csv

5. Run the pipeline
From the project root:
python -m src.pipelines.longitudinal_run



To run rule_agent.py:

1. Create a virtual environment
mac and linux: python3 -m venv venv
windows: python -m venv venv

2. Activate Virtual Environment
mac and linux: source venv/bin/activate
windows (Command Prompt): venv\Scripts\activate
windows (PowerShell): venv\Scripts\Activate.ps1
After activation you should see “(venv)” at the beginning of the terminal line.

3. Install required packages
pip install -r requirements.txt

4. Run the rule based agent
From the project root directory run:
python src/agent/rule_agent.py