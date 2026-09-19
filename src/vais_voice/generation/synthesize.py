"""Reserved for future consented KZ/RU VoiceGen; no synthesis or download occurs."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.parse_args()
    parser.exit(
        2, "VoiceGen backend is not selected. No model downloaded and no audio generated.\n"
    )


if __name__ == "__main__":
    main()
