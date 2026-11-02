"""Elite archive of past policies + opponent sampling for self-play."""
import random


class Archive:
    """Best checkpoints seen so far (across generations), used as opponents and distillation teachers."""

    def __init__(self, size=8, entries=None):
        self.size = size
        self.entries = list(entries or [])  # dicts: path, score, gen, member, stage, hidden

    def add(self, entries):
        self.entries.extend(entries)
        # rank by stage first (harder curriculum = more capable), then score
        self.entries.sort(key=lambda e: (e["stage"], e["score"]), reverse=True)
        del self.entries[self.size:]

    def top(self, k):
        return self.entries[:k]

    def to_list(self):
        return list(self.entries)


def choose_opponent(rng: random.Random, archive: Archive, current, archive_prob=0.5):
    """Pick a frozen opponent checkpoint path: an archived elite or a current population member."""
    pool_a = [e["path"] for e in archive.entries]
    pool_c = [p for p in current if p]
    if not pool_a and not pool_c:
        return None
    if pool_a and (not pool_c or rng.random() < archive_prob):
        return rng.choice(pool_a)
    return rng.choice(pool_c)
