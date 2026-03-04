# twintumor
TwinTumor – Digital Twins for Tumor Growth and Treatment Simulation

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