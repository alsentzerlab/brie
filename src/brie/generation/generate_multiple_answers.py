"""Generate multiple grounded QA variants for each input question.

This explicit entry point exposes the multi-answer augmentation pipeline that
historically lived under :mod:`brie.generation.generate_answer`.
"""

from .generate_answer import main, parse_args


if __name__ == "__main__":
    main(parse_args())
