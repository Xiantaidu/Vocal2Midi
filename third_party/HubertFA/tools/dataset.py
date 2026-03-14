import pathlib

import h5py
import numpy as np
import torch


class IndexedDatasetBuilder:
    def __init__(self, path, prefix, aug_num: int = 1):
        self.path = pathlib.Path(path) / f'{prefix}.data'
        self.prefix = prefix
        self.dset = h5py.File(self.path, 'w')
        self.h5_meta = self.dset.create_group("meta_data")
        self.items = self.dset.create_group("items")
        self.aug_num = aug_num

        self.counter = 0
        self.aug_wav_lengths = []

    def add_item(self, item):
        if item is None:
            return
        group = self.items.create_group(str(self.counter))
        self.counter += 1
        for key, value in item.items():
            if key in ['name', 'ph_seq', 'ph_seq_raw']:
                group.create_dataset(key, data=value, dtype=h5py.string_dtype())
            else:
                group[key] = value
        self.aug_wav_lengths.append(item["wav_length"][0] * self.aug_num)

    def finalize(self):
        self.h5_meta.create_dataset("aug_wav_lengths", data=np.array(self.aug_wav_lengths, dtype=np.float32))
        self.dset.close()


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, binary_data_folder="data/binary", prefix="train"):
        # do not open hdf5 here
        self.h5py_file = None
        self.aug_wav_lengths = None
        self.augmentation_indexes = None

        self.binary_data_folder = binary_data_folder
        self.prefix = prefix

    def get_aug_wav_lengths(self):
        uninitialized = self.aug_wav_lengths is None
        if uninitialized:
            self._open_h5py_file()
        ret = self.aug_wav_lengths
        if uninitialized:
            self._close_h5py_file()
        return ret

    def _open_h5py_file(self):
        self.h5py_file = h5py.File(
            str(pathlib.Path(self.binary_data_folder) / (self.prefix + ".data")), "r"
        )
        self.aug_wav_lengths = np.array(self.h5py_file["meta_data"]["aug_wav_lengths"])

    def _close_h5py_file(self):
        self.h5py_file.close()
        self.h5py_file = None

    def __len__(self):
        uninitialized = self.h5py_file is None
        if uninitialized:
            self._open_h5py_file()
        ret = len(self.h5py_file["items"])
        if uninitialized:
            self._close_h5py_file()
        return ret

    def __getitem__(self, index):
        pass


class NonLexicalLabelerDataset(BaseDataset):
    def __init__(self, binary_data_folder="data/binary", prefix="train"):
        super().__init__(binary_data_folder, prefix)

    def __getitem__(self, index):
        if self.h5py_file is None:
            self._open_h5py_file()
        item = self.h5py_file["items"][str(index)]
        name = [name.decode('utf-8') for name in item["name"]]
        input_feature = np.array(item["input_feature"])  # [1,256,T]
        mel_spec = np.array(item["mel_spec"])
        non_lexical_target = np.array(item["non_lexical_target"])
        non_lexical_intervals = np.array(item["non_lexical_intervals"])
        return name, mel_spec, input_feature, non_lexical_target, non_lexical_intervals


class ForcedAlignmentDataset(BaseDataset):
    def __init__(self, binary_data_folder="data/binary", prefix="train"):
        super().__init__(binary_data_folder, prefix)

    def __getitem__(self, index):
        if self.h5py_file is None:
            self._open_h5py_file()
        item = self.h5py_file["items"][str(index)]
        name = [name.decode('utf-8') for name in item["name"]]
        input_feature = np.array(item["input_feature"])  # [1,256,T]
        ph_seq_raw = [ph.decode('utf-8') for ph in item["ph_seq_raw"]]
        ph_seq = [[ph.decode('utf-8') for ph in phs] for phs in item["ph_seq"]]
        ph_id_seq = np.array(item["ph_id_seq"])
        ph_edge = np.array(item["ph_edge"])
        ph_frame = np.array(item["ph_frame"])
        ph_mask = np.array(item["ph_mask"])
        melspec = np.array(item["mel_spec"])
        ph_time = np.array(item["ph_time"])
        ph_time_raw = np.array(item["ph_time_raw"])
        curves = np.array(item["curves"])
        wav_length = np.array(item["wav_length"])
        return input_feature, ph_seq, ph_id_seq, ph_edge, ph_frame, ph_mask, melspec, ph_time, name, ph_seq_raw, ph_time_raw, curves, wav_length


class BinningAudioBatchSampler(torch.utils.data.Sampler):
    def __init__(
            self,
            wav_lengths: list[int],
            max_length: int = 100,
            binning_length: int = 1000,
            drop_last: bool = False
    ):
        super().__init__()
        assert len(wav_lengths) > 0
        assert max_length > 0
        assert binning_length > 0

        self.max_length = max_length
        self.drop_last = drop_last

        sorted_indices = np.argsort(wav_lengths)[::-1]
        sorted_lengths = np.array(wav_lengths)[sorted_indices]

        self.bins = []
        curr_bin_start_index = 0
        curr_bin_max_length = sorted_lengths[0]

        for i in range(1, len(sorted_lengths)):
            bin_size = i - curr_bin_start_index
            bin_capacity = curr_bin_max_length * bin_size

            if bin_capacity > binning_length:
                batch_size = min(
                    max(1, int(self.max_length // curr_bin_max_length)),
                    bin_size
                )

                bin_indices = sorted_indices[curr_bin_start_index:i]

                if self.drop_last:
                    num_batches = bin_size // batch_size
                else:
                    num_batches = (bin_size + batch_size - 1) // batch_size

                if num_batches > 0:
                    self.bins.append({
                        "indices": bin_indices,
                        "batch_size": batch_size,
                        "num_batches": num_batches
                    })

                curr_bin_start_index = i
                curr_bin_max_length = sorted_lengths[i]

        if curr_bin_start_index < len(sorted_lengths):
            bin_size = len(sorted_lengths) - curr_bin_start_index
            batch_size = min(
                max(1, int(self.max_length // curr_bin_max_length)),
                bin_size
            )

            bin_indices = sorted_indices[curr_bin_start_index:]

            if self.drop_last:
                num_batches = bin_size // batch_size
            else:
                num_batches = (bin_size + batch_size - 1) // batch_size

            if num_batches > 0:
                self.bins.append({
                    "indices": bin_indices,
                    "batch_size": batch_size,
                    "num_batches": num_batches
                })

        self.total_batches = sum(bin_info["num_batches"] for bin_info in self.bins)

    def __iter__(self):
        all_batches = []
        for bin_info in self.bins:
            indices = bin_info["indices"]
            batch_size = bin_info["batch_size"]
            num_batches = bin_info["num_batches"]

            np.random.shuffle(indices)

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                all_batches.append(indices[start_idx:end_idx].copy())

        np.random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        return self.total_batches


def pad_2d(x, target_length, dim=-1):
    return torch.nn.functional.pad(torch.as_tensor(x), (0, target_length - x.shape[dim]), mode='constant', value=0)


def non_lexical_labeler_collate_fn(batch):
    """Collate function for processing a batch of data samples.

    Args:
        batch (list of tuples): Each tuple contains elements from MixedDataset:
            input_feature, ph_seq, ph_edge, ph_frame, ph_mask, melspec.

    Returns:
        input_feature: (B T C)
        input_feature_lengths: (B)
        ph_seq: (B S)
        ph_seq_lengths: (B)
        ph_edge: (B T)
        ph_frame: (B T)
        ph_mask: (B vocab_size)
        mel_spec: (B T)
    """
    # Calculate maximum lengths for padding
    input_feature_lengths = torch.tensor([item[2].shape[-1] for item in batch for _ in range(len(batch[0][2]))])
    max_len = input_feature_lengths.max().item()

    padded_batch = []
    for item in batch:
        padded_batch.append((
            item[0],  # name
            torch.nn.functional.pad(
                torch.as_tensor(item[1]), (0, max_len - item[1].shape[-1]), mode='constant', value=0
            ),  # mel_spec
            torch.nn.functional.pad(
                torch.as_tensor(item[2]), (0, max_len - item[2].shape[-1], 0, 0, 0, 0), mode='constant', value=0
            ),  # input_feature
            pad_2d(item[3], max_len),  # non_lexical_target
            item[4],  # non_lexical_interval
        ))
    return (
        [x_element for x in padded_batch for x_element in x[0]],  # names
        torch.cat([x[1] for x in padded_batch], dim=0),  # mel_specs (B, C_mel, T)
        torch.cat([x[2] for x in padded_batch], dim=0),  # input_features (B, T, C)
        input_feature_lengths,
        torch.cat([x[3] for x in padded_batch], dim=0),  # non_lexical_target (B, N, T)
        [x_element for x in padded_batch for x_element in x[4]],  # non_lexical_intervals (B, N, T)
    )


def forced_alignment_collate_fn(batch):
    """Collate function for processing a batch of data samples.

    Args:
        batch (list of tuples): Each tuple contains elements from MixedDataset:
            input_feature, ph_seq, ph_edge, ph_frame, ph_mask, melspec.

    Returns:
        input_feature: (B T C)
        input_feature_lengths: (B)
        ph_seq: (B S)
        ph_seq_lengths: (B)
        ph_edge: (B T)
        ph_frame: (B T)
        ph_mask: (B vocab_size)
        melspec: (B T)
    """
    # Calculate maximum lengths for padding
    input_feature_lengths = torch.tensor([item[0].shape[-1] for item in batch for _ in range(len(batch[0][0]))])
    max_len = input_feature_lengths.max().item()
    ph_seq_lengths = torch.tensor([len(item[1][0]) for item in batch for _ in range(len(batch[0][0]))])
    max_ph_seq_len = ph_seq_lengths.max().item()

    padded_batch = []
    for item in batch:
        padded_batch.append((
            torch.nn.functional.pad(
                torch.as_tensor(item[0]), (0, max_len - item[0].shape[-1], 0, 0, 0, 0), mode='constant', value=0
            ),  # input_feature
            item[1],  # ph_seq
            pad_2d(item[2], max_ph_seq_len),  # ph_id_seq
            pad_2d(item[3], max_len),  # ph_edge
            pad_2d(item[4], max_len),  # ph_frame
            torch.as_tensor(item[5]),  # ph_mask
            torch.nn.functional.pad(
                torch.as_tensor(item[6]), (0, max_len - item[6].shape[-1]), mode='constant', value=0
            ),  # mel_spec
            pad_2d(item[7], max_ph_seq_len),  # ph_time
            item[8],  # name
            item[9],  # ph_seq_raw
            item[10],  # ph_time_raw
            torch.nn.functional.pad(
                torch.as_tensor(item[11]), (0, max_len - item[11].shape[-1], 0, 0, 0, 0), mode='constant', value=0
            ),  # curves
            torch.as_tensor(item[12])  # wav_length
        ))
    return (
        torch.cat([x[0] for x in padded_batch], dim=0),  # input_features (B, C, T)
        input_feature_lengths,
        [x_element for x in padded_batch for x_element in x[1]],  # ph_seqs
        torch.cat([x[2] for x in padded_batch], dim=0),  # ph_id_seqs (B, S_ph)
        ph_seq_lengths,
        torch.cat([x[3] for x in padded_batch], dim=0),  # ph_edges (B, T)
        torch.cat([x[4] for x in padded_batch], dim=0),  # ph_frames (B, T)
        torch.cat([x[5] for x in padded_batch], dim=0),  # ph_mask (B, ...)
        torch.cat([x[6] for x in padded_batch], dim=0),  # mel_specs (B, C_mel, T)
        torch.cat([x[7] for x in padded_batch], dim=0),  # ph_times (B, S_ph)
        [x_element for x in padded_batch for x_element in x[8]],  # names
        [x[9] for x in padded_batch],  # ph_seq_raws,
        [x[10] for x in padded_batch],  # ph_time_raws
        torch.cat([x[11] for x in padded_batch], dim=0),  # curves
        torch.cat([x[12] for x in padded_batch], dim=0)  # wav_length
    )
