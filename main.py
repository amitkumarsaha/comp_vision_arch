from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    print("Assignment 2 workspace ready.")
    print(f"See {root / 'README.md'} for setup and training commands.")


if __name__ == "__main__":
    main()
