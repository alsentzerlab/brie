"""
One-time precomputation of Octen embeddings for all patients.

Scans --notes for *_subsetrecords.json files, skips patients whose
embeddings are already cached in --embeddings-dir, and encodes the rest
in one large batched model.encode() call.

Usage:
    python precompute_embeddings.py -n <notes_dir> -e <embeddings_dir>
"""

import argparse
import json
import logging
import os

from .rag_utils import build_embeddings_batch, load_cached_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    note_files = [f for f in os.listdir(args.notes) if f.endswith("_subsetrecords.json")]
    if not note_files:
        logger.error(f"No *_subsetrecords.json files found in {args.notes}")
        return

    logger.info(f"Found {len(note_files)} patients in {args.notes}")
    os.makedirs(args.embeddings_dir, exist_ok=True)

    uncached: dict[str, list[dict]] = {}
    for fname in sorted(note_files):
        patient_id = fname.replace("_subsetrecords.json", "")
        if load_cached_embeddings(patient_id, args.embeddings_dir) is not None:
            continue
        with open(os.path.join(args.notes, fname)) as f:
            uncached[patient_id] = json.load(f)

    if not uncached:
        logger.info("All embeddings already cached — nothing to do.")
        return

    logger.info(f"Encoding {len(uncached)} patients ({len(note_files) - len(uncached)} already cached) …")
    build_embeddings_batch(uncached, args.embeddings_dir)
    logger.info(f"Done. Embeddings written to {args.embeddings_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Octen embeddings for all patients")
    parser.add_argument("-n", "--notes", required=True, help="Directory containing *_subsetrecords.json files")
    parser.add_argument("-e", "--embeddings-dir", required=True, help="Directory to write embedding cache files")
    main(parser.parse_args())
