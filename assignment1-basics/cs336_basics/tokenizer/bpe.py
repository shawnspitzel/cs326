import regex as re
import pickle
from pathlib import Path
from collections import Counter


class BPETokenizer:
    def __init__(self, special_tokens: list[str] = []):
        self.special_tokens = special_tokens
        self.vocabulary: dict[bytes, int] = {}
        self.reverseVocab: dict[int, bytes] = {}
        self.corpus: None | str = None
        self.merges: list[tuple[bytes, bytes]] = []
        self.tokenizedCorpus: list = []
        self.sorted_merges = {}
        self._initialize_vocabulary()

    def _initialize_training(self, input_path: str, vocab_size:int):
        self.input_path = input_path
        self.vocab_size = vocab_size
        self._read_corpus()
        self._pre_tokenize_corpus()

    def _read_corpus(self):
        with open(self.input_path, "r", encoding="utf-8") as f:
            self.corpus = f.read()

    def _get_cache_path(self, input_path, vocab_size):
        base = Path(input_path)
        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        st_tag = f"st{len(self.special_tokens or [])}"
        return cache_dir / f"{base.stem}_v{vocab_size}_{st_tag}_cache.pkl"
    
    def _save(self, cache_path: Path):
        with open(cache_path, "wb") as f:
                pickle.dump({
                    "vocabulary": self.vocabulary,
                    "reverseVocab": self.reverseVocab,
                    "merges": self.merges,
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.special_tokens,
                }, f)

    def _load(self, cache_path: Path):
        if not cache_path.exists():
            return False
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            self.vocabulary = data["vocabulary"]
            self.reverseVocab = data["reverseVocab"]
            self.merges = data["merges"]
            self.vocab_size = data["vocab_size"]
            self.special_tokens = data["special_tokens"]

            self.tokenizedCorpus = []
            self.corpus = None
        self.sorted_merges = {
            (a, b): i
            for i, (a, b) in enumerate(self.merges)
        }
        return True

    def _pre_tokenize_corpus(self):
        self.tokenizedCorpus = self._pre_tokenize(self.corpus)

    def _pre_tokenize(self, input_str: str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if type(input_str) is not str:
            raise ValueError("Input not found")
        if not self.special_tokens:
            chunks = [input_str]
            special_set = set()
        else:
            split_pat = f"({'|'.join(map(re.escape, self.special_tokens))})"
            chunks = re.split(split_pat, input_str)
            special_set = set(self.special_tokens)

        encoded_text = []
        for chunk in chunks:
            if chunk in special_set:
                token_bytes = chunk.encode("utf-8")
                token_id = self.vocabulary[token_bytes]
                encoded_text.append([token_id])
                continue
            tokens = re.findall(PAT, chunk)
            # Convert byte values to token IDs using the vocabulary
            for tok in tokens:
                byte_values = tok.encode("utf-8")
                token_ids = [self.vocabulary[bytes([b])] for b in byte_values]
                encoded_text.append(token_ids)

        return encoded_text

    def _initialize_vocabulary(self):
        self.vocabulary = {bytes([x]): x for x in range(256)}
        n = len(self.vocabulary)
        for token in self.special_tokens:
            self.vocabulary[token.encode("utf-8")] = n
            n+=1
        self.reverseVocab = {v: k for k, v in self.vocabulary.items()}

    def _merge_word(self, word, merge_pair, merged_id):
        """Apply a single merge to a word tuple."""
        if len(word) < 2:
            return word

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == merge_pair[0] and word[i + 1] == merge_pair[1]:
                new_word.append(merged_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def _update_pairs_after_merge(self, word_freqs, best_pair, next_id):
        """Incrementally update word frequencies and return only changed pairs."""
        new_word_freqs = {}
        pair_delta = Counter()

        for word, freq in word_freqs.items():
            # Check if this word contains the pair to merge
            has_pair = False
            for i in range(len(word) - 1):
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    has_pair = True
                    break

            if not has_pair:
                # Word unchanged, keep as-is
                new_word_freqs[word] = freq
                continue

            # Remove old pairs from this word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_delta[pair] -= freq

            # Apply merge
            new_word = self._merge_word(word, best_pair, next_id)
            new_word_freqs[new_word] = freq

            # Add new pairs from merged word
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_delta[pair] += freq

        return new_word_freqs, pair_delta

    def _merge_bpe(self, input_enc: list):
        output = []
        for word in input_enc:
            if len(word) < 2:
                output.append(word)
                continue

            word = word[:]
            # Repeatedly find and apply the highest-priority merge
            while True:
                best_rank = float('inf')
                best_pos = None
                best_merged_id = None

                # Scan through the word to find the highest-priority mergeable pair
                for i in range(len(word) - 1):
                    left_id = word[i]
                    right_id = word[i + 1]

                    left_bytes = self.reverseVocab[left_id]
                    right_bytes = self.reverseVocab[right_id]
                    pair = (left_bytes, right_bytes)

                    if pair in self.sorted_merges:
                        rank = self.sorted_merges[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pos = i
                            # Pre-compute merged_id to avoid bytes concatenation in inner loop
                            merged_bytes = left_bytes + right_bytes
                            best_merged_id = self.vocabulary[merged_bytes]

                if best_pos is None:
                    # No more merges possible
                    break

                # Apply the best merge found
                word[best_pos] = best_merged_id
                word.pop(best_pos + 1)

            output.append(word)
        return output
    def train_bpe(self, input_path: str, vocab_size:int):
        cache_path = self._get_cache_path(input_path, vocab_size)

        self.input_path = input_path
        self.vocab_size = vocab_size

        if self._load(cache_path):
            print(f"Loaded tokenizer from cache: {cache_path}")
            return self.reverseVocab, self.merges

        self._initialize_vocabulary()
        self._initialize_training(input_path, vocab_size)

        word_freqs = Counter()
        for word in self.tokenizedCorpus:
            word_freqs[tuple(word)] += 1

        # Build initial pair frequencies once
        pair_freqs = Counter()
        for word, count in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += count

        next_id = len(self.vocabulary)
        while len(self.vocabulary) < vocab_size:
            if not pair_freqs:
                break

            # Find best pair
            best_pair = max(
                pair_freqs.items(),
                key=lambda x: (x[1], (self.reverseVocab[x[0][0]], self.reverseVocab[x[0][1]]))
            )[0]

            # Convert to bytes only once for vocabulary
            merged_bytes = self.reverseVocab[best_pair[0]] + self.reverseVocab[best_pair[1]]
            if merged_bytes in self.vocabulary:
                # Remove this pair and continue
                del pair_freqs[best_pair]
                continue

            self.vocabulary[merged_bytes] = next_id
            self.reverseVocab[next_id] = merged_bytes
            self.merges.append((self.reverseVocab[best_pair[0]], self.reverseVocab[best_pair[1]]))

            # Incremental update: only update affected words and pairs
            word_freqs, pair_delta = self._update_pairs_after_merge(word_freqs, best_pair, next_id)

            # Apply delta to pair frequencies
            for pair, delta in pair_delta.items():
                pair_freqs[pair] += delta
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]

            next_id += 1

        self.sorted_merges = {
            (a, b): i
            for i, (a, b) in enumerate(self.merges)
        }
        self._save(cache_path)
        return self.reverseVocab, self.merges
    
    def encode(self, input_str: str):
        tokenized_str = self._pre_tokenize(input_str)
        encoded_str = self._merge_bpe(tokenized_str)
        # Flatten the list of word-lists into a single list of token IDs
        return [token for word in encoded_str for token in word]

    def decode(self, tokenized_input: list[int]):
        # Collect all bytes first, then decode
        all_bytes = b"".join(self.reverseVocab[token] for token in tokenized_input)
        return all_bytes.decode("utf-8", errors="replace")