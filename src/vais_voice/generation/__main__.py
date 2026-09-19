import sys

from vais_voice.generation import GenerationNotImplementedError, generate

try:
    generate()
except GenerationNotImplementedError as exc:
    sys.exit(str(exc))
