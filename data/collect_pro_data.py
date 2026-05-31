"""
Drop raw professional videos into:

data/professional/forehand/
data/professional/backhand/
data/professional/serve/

Then run:

python data/collect_pro_data.py
python data/collect_pro_data.py --stroke forehand
python data/collect_pro_data.py --force
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.reference_builder import build_all_references, build_reference_for_stroke


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stroke', choices=['forehand', 'backhand', 'serve'], default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--target-len', type=int, default=120)
    args = parser.parse_args()

    base_prof_dir = os.path.join(PROJECT_ROOT, 'data', 'professional')

    if args.stroke:
        build_reference_for_stroke(
            base_prof_dir,
            args.stroke,
            target_len=args.target_len,
            verbose=True
        )
    else:
        build_all_references(
            base_prof_dir,
            strokes=('forehand', 'backhand', 'serve'),
            target_len=args.target_len,
            force=args.force
        )


if __name__ == '__main__':
    main()
