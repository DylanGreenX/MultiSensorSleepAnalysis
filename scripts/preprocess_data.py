"""Preprocessing data CLI wrapper."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multisensor_sleep.preprocessing.preprocess import main

if __name__ == "__main__":
    main()