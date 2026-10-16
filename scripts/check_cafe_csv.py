import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.csv_validation import validate_cafe_csv

def main():
    parser = argparse.ArgumentParser(description='Check numeric cafe CSV input')
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    try:
        rows = validate_cafe_csv(args.input)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(dict(valid=True, rows=rows)))

if __name__ == '__main__':
    main()
