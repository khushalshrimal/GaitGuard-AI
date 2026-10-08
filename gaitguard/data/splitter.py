"""
GaitGuard AI - Group-Aware Data Splitter Module (Phase 2)
Generates leak-free cross-validation and test splits grouped by animal_id.
"""

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

class GroupDataSplitter:
    def __init__(self, n_splits=5, seed=42):
        self.n_splits = n_splits
        self.seed = seed

    def generate_kfold_splits(self, animal_ids):
        """
        Generates 5-fold GroupKFold indices.
        Guarantees 0 animal overlap between train and validation folds.
        """
        animal_ids = np.asarray(animal_ids)
        dummy_x = np.zeros(len(animal_ids))
        
        gkf = GroupKFold(n_splits=self.n_splits)
        splits = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(dummy_x, groups=animal_ids)):
            train_animals = set(animal_ids[train_idx])
            val_animals = set(animal_ids[val_idx])
            overlap = train_animals.intersection(val_animals)
            
            if len(overlap) > 0:
                raise ValueError(f"Data Leakage Error in fold {fold_idx}: Overlapping animals {overlap}")
            
            splits.append({
                'fold': fold_idx,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_animals_count': len(train_animals),
                'val_animals_count': len(val_animals),
                'train_samples_count': len(train_idx),
                'val_samples_count': len(val_idx)
            })
            
        return splits

    def generate_train_val_test_split(self, animal_ids, test_size=0.2, val_size=0.2):
        """
        Generates a 3-way GroupShuffleSplit (Train / Val / Test).
        Guarantees 0 animal overlap across Train, Val, and Test sets.
        """
        animal_ids = np.asarray(animal_ids)
        indices = np.arange(len(animal_ids))
        
        # 1. Split off Test set
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
        train_val_idx, test_idx = next(gss_test.split(indices, groups=animal_ids))

        # 2. Split remaining into Train and Validation
        relative_val_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=self.seed)
        train_sub_idx, val_sub_idx = next(gss_val.split(train_val_idx, groups=animal_ids[train_val_idx]))
        
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]

        train_cows = set(animal_ids[train_idx])
        val_cows = set(animal_ids[val_idx])
        test_cows = set(animal_ids[test_idx])

        # Verify zero leakage
        assert len(train_cows.intersection(val_cows)) == 0, "Train-Val Leakage!"
        assert len(train_cows.intersection(test_cows)) == 0, "Train-Test Leakage!"
        assert len(val_cows.intersection(test_cows)) == 0, "Val-Test Leakage!"

        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'train_cows': sorted(list(train_cows)),
            'val_cows': sorted(list(val_cows)),
            'test_cows': sorted(list(test_cows))
        }
