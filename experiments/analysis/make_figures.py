"""Every table and figure of the paper, from saved results only. The plain-Python "make figures".

  python experiments/analysis/make_figures.py              reads results/, writes paper/tables and paper/figures
  python experiments/analysis/make_figures.py --stand-in   the pipeline test: reads and writes results/standin_not_results

Each script reads only saved files and plays no episode. A table whose scores do not exist yet is
written with "--" in those cells, so the paper always builds.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = ["table_rules.py", "table_belief.py", "table_main.py", "table_tuning.py", "table_ablations.py",
           "table_mistakes.py", "fig_curves.py"]

if __name__ == "__main__":
    for script in SCRIPTS:
        print(f"== {script}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *sys.argv[1:]], check=True)
