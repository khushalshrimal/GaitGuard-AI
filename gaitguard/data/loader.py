"""
GaitGuard AI - Raw Dataset Loader Module (Phase 2)
Locates and loads raw keypoint trajectory CSVs and ground-truth scores.
"""

import os
import glob
import pandas as pd
import numpy as np

KEYPOINT_NAMES = [
    'LFHoof', 'LFAnkle', 'LFKnee',
    'RFHoof', 'RFAnkle', 'RFKnee',
    'LHHoof', 'LHAnkle', 'LHKnee',
    'RHHoof', 'RHAnkle', 'RHKnee',
    'Nose', 'HeadTop',
    'Spine1', 'Spine2', 'Spine3'
]

class RawDataLoader:
    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self.russello_base = os.path.join(
            self.root_dir, "Datasets", "lstm-lameness-detection-main", "lstm-lameness-detection-main"
        )
        self.kp_dir = os.path.join(self.russello_base, "data", "videos_keypoints")
        self.scores_file = os.path.join(self.russello_base, "data", "videos_lameness_scores.csv")

    def load_scores(self):
        if not os.path.exists(self.scores_file):
            raise FileNotFoundError(f"Scores file missing at {self.scores_file}")
        scores_df = pd.read_csv(self.scores_file, dtype={'Video': str, 'ID': int, 'hard_vote': int})
        return scores_df

    def load_sample_keypoints(self, video_id):
        kp_path = os.path.join(self.kp_dir, f"{video_id}.csv")
        if not os.path.exists(kp_path):
            raise FileNotFoundError(f"Keypoint CSV file missing for video {video_id} at {kp_path}")
        df = pd.read_csv(kp_path)
        return df

    def load_all_raw_samples(self):
        scores_df = self.load_scores()
        raw_samples = []

        for idx, row in scores_df.iterrows():
            video_id = str(row['Video']).zfill(3)
            animal_id = int(row['ID'])
            hard_vote = int(row['hard_vote'])

            kp_df = self.load_sample_keypoints(video_id)
            
            # Extract 17 keypoint triples (x, y, likelihood) in canonical order
            kp_matrix = []
            for kp in KEYPOINT_NAMES:
                x_col = f"{kp}_x"
                y_col = f"{kp}_y"
                like_col = f"{kp}_likelihood"

                if x_col not in kp_df.columns or y_col not in kp_df.columns:
                    raise KeyError(f"Missing column for keypoint {kp} in video {video_id}")
                
                # Default likelihood to 1.0 if likelihood column missing
                l_vals = kp_df[like_col].values if like_col in kp_df.columns else np.ones(len(kp_df))
                
                triple = np.column_stack([kp_df[x_col].values, kp_df[y_col].values, l_vals])
                kp_matrix.append(triple)

            # Transpose to shape (T, 17, 3)
            kp_array = np.stack(kp_matrix, axis=1) # (T, 17, 3)

            sample = {
                'sample_id': video_id,
                'animal_id': animal_id,
                'video_id': video_id,
                'raw_sequence_length': len(kp_df),
                'raw_keypoints': kp_array, # shape (T, 17, 3)
                'raw_label': hard_vote
            }
            raw_samples.append(sample)

        return raw_samples
