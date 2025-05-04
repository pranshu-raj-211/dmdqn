# Multi Agent Deep Q Networks for Traffic Signal Optimization

## How to run

1. Set pythonpath (avoids errors like no module named src).
    `export PYTHONPATH=/path/to/dmdqn`

    or simply:
    ```export PYTHONPATH=`pwd` ```
2. Download all dependencies
   
   If using uv:

   `uv sync`

   If using pip:

   ```bash
   pip install pip-tools
   python -m piptools compile \\
    -o requirements.txt \\
    pyproject.toml
   pip install -r requirements.txt
   ```
3. Run src/scripts/train.py
    `python src/scripts/train.py`


### Common errors and how to fix them

1. **Wandb init error**: Usually occurs due to a bad internet connection, try pinging api.wandb.ai first, if that succeeds, it is probably a DNS issue.
   
   `ping api.wandb.ai`   -> Passes

   `nslookup api.wandb.ai` -> Fails

   Then it is a DNS issue

   Fix using:

   `sudo nano etc/resolv.conf`

   Inside nano, right after the comments, write:

   `nameserver 8.8.8.8`
   
   or use other common nameservers like 8.8.4.4 or 1.1.1.1

2. Ignore warnings for emergency braking (disable by disabling stderr in sumo cfg options)
3. 