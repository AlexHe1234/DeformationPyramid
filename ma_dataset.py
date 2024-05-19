import numpy as np
from torch.utils import data
from typing import Literal, List
from SceneTracker.lib.dataset.util import *
import json


class MixamoAMASS(data.Dataset):
    def __init__(self,
                 split: Literal['train', 'val', 'test'],  # passed in by main.py

                 root_dir: str,
                 num_tracks: int=256,
                 num_points: int=2048,
                 max_frame: int=50,

                 rand_rotation: bool=True,
                 rand_rotate_z_only: bool=True,
                 rand_padding: bool=True,
                 rand_padding_range: List[float]=[0.05, 0.1],
                 rand_reordering: bool=False,
                 rand_perturb: bool=True,
                 rand_perturb_avg_sd: List[float]=[0., 1.e-5],

                 uni_prob: float=1.,
                 noise_ratio: float=0.001,
                 fps_method: str='fpsample',

                 **kwargs,
                 ):
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Split type {split} is not supported'
        self.split = split
        if self.split != 'train':
            np.random.seed(42)

        self.num_tracks = num_tracks
        self.num_points = num_points
        self.max_frame = max_frame

        self.rand_rotation = rand_rotation
        self.rand_rotate_z_only = rand_rotate_z_only
        self.rand_padding = rand_padding
        self.rand_padding_range = rand_padding_range
        self.rand_reordering = rand_reordering
        self.rand_perturb = rand_perturb
        self.rand_perturb_avg_sd = rand_perturb_avg_sd
        self.uni_prob = uni_prob
        self.noise_ratio = noise_ratio
        self.fps_method = fps_method

        self.root_dir = root_dir
        if self.split == 'train':
            sf = os.path.join(self.root_dir, '0train.json')
        elif self.split == 'val':
            sf = os.path.join(self.root_dir, '0val.json')
        else:
            sf = os.path.join(self.root_dir, '0test.json')
            
        with open(sf, 'r') as f:
            seqs = json.load(sf)
        
        self.seqs = seqs
        # seqs = sorted([f for f in os.listdir(self.root_dir) if f[-3:] == 'npy'])
        # if self.split == 'train':
        #     self.seqs = seqs[2:]
        # elif self.split == 'val':
        #     self.seqs = seqs[:2]
        # else:
        #     raise ValueError(f'Split type {self.split} is not supported')

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        seq = np.load(os.path.join(self.root_dir, seq_name))  # F_og, P_og, 3

        nf = seq.shape[0]
        if nf > self.max_frame:
            start_frame = np.random.randint(0, nf - self.max_frame + 1)
            seq = seq[start_frame : start_frame + self.max_frame]
            nf = seq.shape[0]

        gt_seq_copy = seq.copy()

        seq, reverse_transform = proc_frame_points(seq,
                                                   ret_reverse_transform=True,
                                                   use_rand_rotation=self.rand_rotation,
                                                   rand_rotate_z_only=self.rand_rotate_z_only,
                                                   use_rand_padding=self.rand_padding,
                                                   rand_padding_range=self.rand_padding_range)

        input_seq = seq.copy()

        if self.rand_reordering:
            input_seq = rand_reorder_by_frame(input_seq)

        if self.rand_perturb:
            input_seq += np.random.randn(*input_seq.shape) * self.rand_perturb_avg_sd[1] + self.rand_perturb_avg_sd[0]

        rand_sample_buffer = np.empty((nf, self.num_points if self.num_points < input_seq.shape[1] else input_seq.shape[1], 3))
        num_point_samples = int(rand_sample_buffer.shape[1] * (1. - self.noise_ratio))
        for i in range(nf):
            seq_i = input_seq[i]  # N, 3
            rot_matrix = Rotation.random().as_matrix().T
            seq_i = seq_i @ rot_matrix
            sampled_i = fps(seq_i, num_point_samples, method=self.fps_method)
            rand_sample_buffer_i = sampled_i @ rot_matrix.T
            rand_sample_buffer[i] = noisify_pcd(rand_sample_buffer_i, rand_sample_buffer.shape[1] - num_point_samples)


        rand_sample_buffer[rand_sample_buffer > 1.] = 1. - 1e-6
        rand_sample_buffer[rand_sample_buffer < -1.] = -1. + 1e-6

        sampled_verts_indices = smart_sample(seq[0], self.num_tracks, self.uni_prob)
        # sampled_verts_indices = np.random.choice(seq.shape[1], size=self.num_tracks, replace=False)
        sampled_verts = seq[:, sampled_verts_indices]

        transform = np.eye(4)[None]
        for tran_mat in reversed(reverse_transform):
            transform = np.matmul(transform, tran_mat)

        ret = {
            'reverse_transform': transform,  # nf, 4, 4
            'tracks': sampled_verts,  # nf, ntrack, 3
            'tracks_og': gt_seq_copy[:, sampled_verts_indices],  # nf, ntrack, 3
            'points': rand_sample_buffer,  # nf, npoint, 3
            'first': np.zeros((sampled_verts.shape[1])),  # TODO: ntrack
        }

        return ret
