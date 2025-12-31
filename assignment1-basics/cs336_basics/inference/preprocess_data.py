import argparse
import numpy as np
import os
from pathlib import Path
from multiprocessing import Pool
from cs336_basics.tokenizer.bpe import BPETokenizer
from cs336_basics.inference.pretokenization import find_chunk_boundaries


def _encode_chunk_worker(args):
    """
    worker function for parallel chunk encoding
    """
    chunk_text, tokenizer_state = args
    tokenizer = BPETokenizer()
    tokenizer.vocabulary = tokenizer_state['vocabulary']
    tokenizer.reverseVocab = tokenizer_state['reverseVocab']
    tokenizer.merges = tokenizer_state['merges']

    return tokenizer.encode(chunk_text)


def preprocess_data(
    train_file: str,
    val_file: str = None,
    vocab_size: int = 50257,
    output_dir: str = None,
    num_workers: int = 4
):
    """
    Tokenize raw text files, save as binary arrays

    Inputs:
        train_file: Path to raw training text file
        val_file: Path to raw validation text file
        vocab_size: Size of BPE vocabulary
        output_dir: Directory to save tokenized binary files
        num_workers: Number of parallel workers for chunked encoding
    """
    script_path = Path(__file__).resolve()
    package_root = script_path.parent.parent

    if output_dir is None:
        output_dir = package_root / 'data/tokenized'
    else:
        output_dir = package_root / output_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Training BPE Tokenizer..")
    print("=" * 70)
    print(f"  Training file: {train_file}")
    print(f"  Vocab size: {vocab_size}")
    print("  Note: Your BPETokenizer caches results automatically in ./cache/")
    print()

    tokenizer = BPETokenizer()
    tokenizer.train_bpe(train_file, vocab_size=vocab_size)

    print("Finished training")
    print(f"  Vocabulary size: {len(tokenizer.vocabulary)}")
    print(f"  Number of merges: {len(tokenizer.merges)}")
    print()

    print("=" * 70)
    print("Tokenizing data..")
    print("=" * 70)
    print(f"  Reading from: {train_file}")
    print(f"  Using {num_workers} parallel workers")

    # find chunk boundaries
    with open(train_file, 'rb') as f:
        train_boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

    chunks = []
    with open(train_file, 'rb') as f:
        for start, end in zip(train_boundaries[:-1], train_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    tokenizer_state = {
        'vocabulary': tokenizer.vocabulary,
        'reverseVocab': tokenizer.reverseVocab,
        'merges': tokenizer.merges
    }

    # encode chunks in parallel
    if num_workers > 1:
        with Pool(num_workers) as pool:
            chunk_tokens_list = pool.map(_encode_chunk_worker, [(chunk, tokenizer_state) for chunk in chunks])
    else:
        chunk_tokens_list = [tokenizer.encode(chunk) for chunk in chunks]

    train_tokens = []
    for i, chunk_tokens in enumerate(chunk_tokens_list):
        train_tokens.extend(chunk_tokens)
        print(f"    Chunk {i+1}/{len(chunks)}: {len(chunk_tokens):,} tokens")

    train_tokens_array = np.array(train_tokens, dtype=np.uint16)

    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    train_basename = os.path.splitext(os.path.basename(train_file))[0]  
    train_output_path = train_dir / f"{train_basename}_tokens_.bin"
    train_tokens_array.tofile(train_output_path)

    # Only process validation file if provided
    if val_file is None:
        print("=" * 70)
        print("Preprocessing complete!")
        print("=" * 70)
        print("\n Created files:")
        print(f"  1. {train_output_path}")
        print(f"  Vocabulary size: {len(tokenizer.vocabulary):,}")
        return

    print(f"\n  Reading from: {val_file}")

    with open(val_file, 'rb') as f:
        val_boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

    print(f"  Split into {len(val_boundaries) - 1} chunks")
    print("  Encoding chunks in parallel...")

    val_chunks = []
    with open(val_file, 'rb') as f:
        for start, end in zip(val_boundaries[:-1], val_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            val_chunks.append(chunk)
            
    if num_workers > 1:
        with Pool(num_workers) as pool:
            val_chunk_tokens_list = pool.map(_encode_chunk_worker, [(chunk, tokenizer_state) for chunk in val_chunks])
    else:
        val_chunk_tokens_list = [tokenizer.encode(chunk) for chunk in val_chunks]

    val_tokens = []
    for i, chunk_tokens in enumerate(val_chunk_tokens_list):
        val_tokens.extend(chunk_tokens)
        print(f"    Chunk {i+1}/{len(val_chunks)}: {len(chunk_tokens):,} tokens")

    val_tokens_array = np.array(val_tokens, dtype=np.uint16)

    val_dir = output_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    val_basename = os.path.splitext(os.path.basename(val_file))[0]  
    val_output_path = val_dir / f"{val_basename}_val_tokens_.bin"
    val_tokens_array.tofile(val_output_path)

    print("=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)
    print("\n Created files:")
    print(f"  1. {train_output_path}")
    print(f"  2. {val_output_path}")
    print("  3. Tokenizer cache (in ./cache/ directory)")
    print("\n Dataset statistics:")
    print(f"  Total training tokens: {len(train_tokens_array):,}")
    print(f"  Total validation tokens: {len(val_tokens_array):,}")
    print(f"  Training file size: {os.path.getsize(train_output_path) / 1024 / 1024:.2f} MB")
    print(f"  Validation file size: {os.path.getsize(val_output_path) / 1024 / 1024:.2f} MB")
    print(f"  Vocabulary size: {len(tokenizer.vocabulary):,}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess text data for training")

    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to raw training text file")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Path to raw validation text file (optional)")
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="BPE vocabulary size (default: 50257 for GPT-2)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save tokenized files (default: cs336_basics/data)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for chunked encoding (default: 4)")

    args = parser.parse_args()

    preprocess_data(
        train_file=args.train_file,
        val_file=args.val_file,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
