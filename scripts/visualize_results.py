"""Visualization CLI script."""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multisensor_sleep.visualization.single_sensor import main as single_main
from multisensor_sleep.visualization.multi_sensor import main as multi_main
from multisensor_sleep.visualization.consensus import main as consensus_main
from multisensor_sleep.visualization.comparison import main as comparison_main

def main():
    parser = argparse.ArgumentParser(
        description="Visualize sleep analysis results"
    )
    parser.add_argument('file', help='Results CSV file to visualize')
    parser.add_argument('-t', '--type',
                       choices=['single', 'multi', 'consensus', 'comparison'],
                       help='Type of visualization')
    parser.add_argument('-o', '--output', default='results/visualizations/',
                       help='Output directory for saved plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Auto-detect visualization type if not specified
    if not args.type:
        # Try to detect from filename or file contents
        if 'single' in args.file:
            args.type = 'single'
        elif 'mult' in args.file or 'multi' in args.file:
            args.type = 'multi'
        elif 'consensus' in args.file:
            args.type = 'consensus'
        else:
            args.type = 'single'  # Default

    # Override sys.argv for the visualization modules
    sys.argv = ['visualize_results.py', args.file]

    print(f"Generating {args.type} visualization for {args.file}")

    if args.type == 'single':
        single_main()
    elif args.type == 'multi':
        multi_main()
    elif args.type == 'consensus':
        consensus_main()
    elif args.type == 'comparison':
        comparison_main()

    print(f"Visualization saved to {args.output}")

if __name__ == "__main__":
    main()