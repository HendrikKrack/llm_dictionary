import os
from typing import List, Tuple

def load_definitions(filepath: str) -> List[Tuple[str, str]]:
    """Load term-definition pairs from the testDefinitions.txt file."""
    definitions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if '[DEF]' in line:
                term, definition = line.split('[DEF]', 1)
                definitions.append((term.strip(), definition.strip()))
    return definitions

if __name__ == "__main__":
    # Example usage
    data = load_definitions(os.path.join('..', 'testData', 'testDefinitions.txt'))
    print(f"Loaded {len(data)} definitions.")
