"""Run openWakeWord's train.py unchanged on macOS.

Its DataLoader passes lambdas to workers, which the default 'spawn' start method cannot
pickle, so force 'fork' (what Linux, the recipe's target, uses).

Usage: python -I run_train.py PATH/TO/train.py <train.py args...>
"""

import multiprocessing
import runpy
import sys

multiprocessing.set_start_method("fork", force=True)
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
