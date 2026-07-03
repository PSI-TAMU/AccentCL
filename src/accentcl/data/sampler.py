import math
import random
from collections import defaultdict
from torch.utils.data import Sampler

class OldNewBalancedBatchSampler(Sampler):
    """
    Creates batches with a fixed old/new ratio using `is_replay`.

    Expected df columns:
      - accent
      - is_replay

    is_replay == True  -> old replay memory samples
    is_replay == False -> new-class samples

    If old_class_balanced=True, the old replay part of each batch is
    approximately class-balanced across old accent classes.
    """

    def __init__(
        self,
        df,
        batch_size: int,
        old_ratio: float = 0.5,
        old_class_balanced: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.old_ratio = old_ratio
        self.old_class_balanced = old_class_balanced
        self.seed = seed
        self.drop_last = drop_last

        if "is_replay" not in self.df.columns:
            raise ValueError("df must contain an `is_replay` column.")

        if "accent" not in self.df.columns:
            raise ValueError("df must contain an `accent` column.")

        self.num_old = int(round(batch_size * old_ratio))
        self.num_new = batch_size - self.num_old

        if self.num_old <= 0 or self.num_new <= 0:
            raise ValueError(
                f"Invalid old/new split: num_old={self.num_old}, num_new={self.num_new}"
            )

        is_replay = self.df["is_replay"].astype(bool).values
        accents = self.df["accent"].astype(str).values

        self.old_indices = [i for i, x in enumerate(is_replay) if x]
        self.new_indices = [i for i, x in enumerate(is_replay) if not x]

        if len(self.old_indices) == 0:
            raise ValueError("No replay/old samples found: is_replay=True.")

        if len(self.new_indices) == 0:
            raise ValueError("No new samples found: is_replay=False.")

        # Group old replay samples by accent label.
        self.old_indices_by_label = defaultdict(list)

        for i in self.old_indices:
            self.old_indices_by_label[accents[i]].append(i)

        self.old_labels = sorted(self.old_indices_by_label.keys())

        if self.old_class_balanced and len(self.old_labels) == 0:
            raise ValueError("old_class_balanced=True, but no old labels found.")

        # Epoch length: define by larger side. Pools wrap around when exhausted.
        if drop_last:
            self.num_batches = max(
                len(self.old_indices) // self.num_old,
                len(self.new_indices) // self.num_new,
            )
        else:
            self.num_batches = max(
                math.ceil(len(self.old_indices) / self.num_old),
                math.ceil(len(self.new_indices) / self.num_new),
            )

        self.rng = random.Random(self.seed)

        print(
            f"OldNewBalancedBatchSampler: "
            f"old={len(self.old_indices)}, new={len(self.new_indices)}, "
            f"old_labels={len(self.old_labels)}, "
            f"batch={batch_size}, num_old={self.num_old}, num_new={self.num_new}, "
            f"old_class_balanced={self.old_class_balanced}, "
            f"num_batches={self.num_batches}"
        )

        if self.old_class_balanced:
            print("Old replay counts by label:")
            for label in self.old_labels:
                print(f"  {label:20s}: {len(self.old_indices_by_label[label])}")

    def _sample_from_pool(self, pool, ptr, n_take):
        """
        Sample n_take items from a shuffled pool.
        If the pool is exhausted, reshuffle and wrap around.
        """
        selected = []

        while len(selected) < n_take:
            remaining = len(pool) - ptr
            need = n_take - len(selected)

            if remaining >= need:
                selected.extend(pool[ptr : ptr + need])
                ptr += need
            else:
                if remaining > 0:
                    selected.extend(pool[ptr:])

                self.rng.shuffle(pool)
                ptr = 0

        return selected, ptr

    def _get_old_label_budgets(self, batch_idx):
        """
        Allocate self.num_old samples across old labels.

        Example:
            num_old = 20, num_labels = 6
            budgets are 4,4,3,3,3,3, rotating which labels get 4.
        """
        n_labels = len(self.old_labels)

        base = self.num_old // n_labels
        remainder = self.num_old % n_labels

        budgets = {label: base for label in self.old_labels}

        # Rotate extra samples across batches so the same labels do not always get +1.
        for r in range(remainder):
            label = self.old_labels[(batch_idx + r) % n_labels]
            budgets[label] += 1

        # If num_old < n_labels, base=0 and only some labels get 1 per batch.
        return budgets

    def __iter__(self):
        # New pool
        new_pool = self.new_indices.copy()
        self.rng.shuffle(new_pool)
        new_ptr = 0

        # Old pools
        if self.old_class_balanced:
            old_pools = {
                label: indices.copy()
                for label, indices in self.old_indices_by_label.items()
            }
            old_ptrs = {label: 0 for label in self.old_labels}

            for label in self.old_labels:
                self.rng.shuffle(old_pools[label])

        else:
            old_pool = self.old_indices.copy()
            self.rng.shuffle(old_pool)
            old_ptr = 0

        for batch_idx in range(self.num_batches):
            # -------------------------
            # Sample old replay part
            # -------------------------
            if self.old_class_balanced:
                old_batch = []
                budgets = self._get_old_label_budgets(batch_idx)

                for label, n_take in budgets.items():
                    if n_take <= 0:
                        continue

                    selected, new_ptr_label = self._sample_from_pool(
                        pool=old_pools[label],
                        ptr=old_ptrs[label],
                        n_take=n_take,
                    )

                    old_ptrs[label] = new_ptr_label
                    old_batch.extend(selected)

            else:
                old_batch, old_ptr = self._sample_from_pool(
                    pool=old_pool,
                    ptr=old_ptr,
                    n_take=self.num_old,
                )

            # -------------------------
            # Sample new-class part
            # -------------------------
            new_batch, new_ptr = self._sample_from_pool(
                pool=new_pool,
                ptr=new_ptr,
                n_take=self.num_new,
            )

            batch = old_batch + new_batch
            self.rng.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches