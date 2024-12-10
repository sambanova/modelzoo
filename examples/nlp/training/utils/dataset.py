# Copyright 2020-2024 SambaNova Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler

#from sambaflow.samba.utils import get_world_size


class HDF5ParallelLoader:
    """Represents an Iterator across Dataloaders.  The purpose of this is to support loading very large datasets
    incrementally, one dataloader at a time.
    """
    DTYPE = np.int32()
    GB_TO_B = 2**30

    def __init__(self,
                 data_dir: str,
                 sequence_size: int,
                 local_batch_size: int = 1,
                 rank: int = 0,
                 world_size: int = 1,
                 seed: int = 42,
                 start_batch_index: int = 0,
                 max_dataset_size: float = 5,
                 is_linear: bool = False,
                 drop_last: bool = False,
                 train: bool = True):
        """Creates the DataLoaderIterator.

        Args:
            data_dir: Path to directory containing all the data files.
            local_batch_size:  Batch Size per Data Parallel (DP) Worker.
            rank:  Data Parallel Rank.
            world_size:  Total number of data parallel workers.
            sequence_size: Sequence Size of the sequences in the dataset.
            seed:  Torch RNG seed.
            start_batch_index:  The index to start iterating from (corresponds to "skip_steps")
            max_dataset_size:  The maximum amount of data that can be loaded into memory.  Defaults to 5 GB.
            is_linear:  Whether to iterate through the dataset linearly or randomly.
            drop_last:  Whether to drop the last batch of the dataloader if the batch is smaller than batch_size.
            train:  Whether to load in training files or not
        """

        # BATCH_SIZE ATTRIBUTES
        self.global_batch_size = local_batch_size * world_size
        self.local_batch_size = local_batch_size

        # DATA ATTRIBUTES
        self.data_dir = Path(data_dir)
        bytes_per_sequence = sequence_size * self.DTYPE.itemsize
        max_dataset_size_bytes = int(max_dataset_size * self.GB_TO_B)
        self.max_dataset_length = max_dataset_size_bytes // bytes_per_sequence
        # make max_dataset_length divisible by global batch size such that
        # there will only be one irregularly sized batch at the very end of the DataLoaderIterator
        self.max_dataset_length = (self.max_dataset_length // self.global_batch_size) * self.global_batch_size
        # validation
        err_msg = f'Max dataset size: {max_dataset_size} too small for global batch size {self.global_batch_size}'
        assert self.max_dataset_length >= self.global_batch_size, err_msg

        self.data_files = self._get_files(seed, is_linear)
        # number of sequences in each data file
        self.data_file_lengths = self._get_data_file_lengths()
        self.data_files_total_len = sum(self.data_file_lengths)

        # ITERATOR ATTRIBUTES
        # class invariant:  self.batch_index < self.length
        self.batch_index = 0
        self.curr_dataloader = None
        self.drop_last = drop_last
        # number of batches in the dataloader
        round_fn = math.floor if drop_last else math.ceil
        self.num_batches = round_fn(self.data_files_total_len / self.global_batch_size)
        self.is_linear = is_linear
        self.seed = seed
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)

        # DATA PARALLEL ATTRIBUTES
        self.batch_start_index = rank * local_batch_size
        self.batch_end_index = self.batch_start_index + local_batch_size
        self.rank = rank
        self.world_size = world_size

    def _get_files(self, seed: int, is_linear: bool, train: bool = True) -> List[Path]:
        """Get the data files from the file directory.

        Args:
            seed:  Seed to control random shuffling for the files in the directory.
            is_linear:  Whether to load the files linearly by their index or randomly shuffle.
            train:  Whether to load in training files or eval files
        Returns:
            The list of data files in the data file directory
        """
        if train:
            data_files = sorted(self.data_dir.glob('*train*'), key=get_data_file_index)
        else:
            eval_files = []
            for file_key in ['dev', 'test', 'val']:
                eval_files.extend(list(self.data_dir.glob(f'*{file_key}*')))
            data_files = sorted(eval_files, key=get_data_file_index)
        if not is_linear:
            random.seed(seed)
            random.shuffle(data_files)
        return data_files

    def _get_data_file_lengths(self) -> List[int]:
        """Get the lengths of each of the files in the data directory."""
        data_file_lengths = []
        for data_file in self.data_files:
            with h5py.File(data_file) as f:
                data_lengths = list(map(len, f.values()))
                # all datasets within a single train file should have the same length
                assert len(set(data_lengths)) == 1, f'Dataset lengths inconsistent in file {data_file}'
                data_file_lengths.append(data_lengths[0])
        return data_file_lengths

    def _init_dataloader(self):
        """Creates the DataLoader around as much data that we can load into memory.  We have the option to randomly
        sample from this dataset by using the RandomSampler.
        """
        # current sequence index
        curr_seq_index = self.batch_index * self.global_batch_size
        num_remaining_sequences = self.data_files_total_len - curr_seq_index
        # Dataset length is defined by number of sequences, dataloader length is defined by number of
        # batches.  This can get confusing to think about so make sure you get a cup of coffee before
        # modifying this code.
        dataset_length = min(self.max_dataset_length, num_remaining_sequences)
        dataset = InMemoryDataset(self.data_files, self.data_file_lengths, curr_seq_index, dataset_length, self.DTYPE)
        sampler = None if self.is_linear else RandomSampler(dataset, generator=self.torch_rng)
        self.curr_dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.global_batch_size)
        # need this to call 'next()' on curr_dataloader_enumerator
        self.curr_dataloader_enumerator = enumerate(self.curr_dataloader)
        err_msg_1 = f'Batch Index: {self.batch_index}; Curr Dataloader Length: {len(self.curr_dataloader)};'
        err_msg_2 = f'Num Batches: {self.num_batches}'
        err_msg = f'{ALGO_BUG_MSG} {err_msg_1} {err_msg_2}'
        # need the '+ 1' for the case when drop_last == True, and the entire dataset fits in memory
        assert self.batch_index + len(self.curr_dataloader) <= self.num_batches + 1, err_msg

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        batch = None

        if self.batch_index == self.num_batches:
            raise StopIteration
        assert self.batch_index < len(self)

        # instantiate the dataloader
        if self.curr_dataloader is None:
            self._init_dataloader()
        try:
            _, batch = next(self.curr_dataloader_enumerator)
            batch = Batch(batch)
        except StopIteration:
            pass

        # Keep looping if we have more data
        if batch is not None:
            self.batch_index += 1
            return batch[self.batch_start_index:self.batch_end_index]

        # create new dataloader as part of recursive call.
        self.curr_dataloader = None
        # don't increment batch_index because that will be done in the recursive call
        return next(self)


ALGO_BUG_MSG_1 = 'There is an algorithmic bug in this code, this should be impossible.'
ALGO_BUG_MSG_2 = 'Please message someone on the NLP team to fix this.'
ALGO_BUG_MSG = f'{ALGO_BUG_MSG_1} {ALGO_BUG_MSG_2}'


def get_data_file_index(path):
    try:
        return int(path.stem.split('_')[1])
    except ValueError as e:
        # if raised, then path is not in the form train_x_of_y.hdf5
        # some older shards have the convention shardx_train.hdf5, this will catch that
        match_obj = re.search(r'shard([0-9]+)', str(path.stem))
        if match_obj is not None:
            return int(match_obj.group(1))
        else:
            try:
                # A special case for the finance dataset, which is used in today's regression test
                return int(path.stem.split('_')[-1])
            except:
                raise e


def get_dataset_metadata(path: str) -> Dict[str, any]:
    """
    Read the metadata.yaml from the dataset folder and return the contents

    Arguments:
    * path: the path to the dataset folder

    Returns:
    * Dict representation of contents
    """
    with open(path + '/metadata.yaml', 'r') as f:
        metadata = yaml.safe_load(f)

    return metadata


class Batch:
    """Represents a batch of data."""
    def __init__(self, data: Dict[str, torch.tensor] = None):
        if data is None:
            data = {}

        self.data = data
        self._update_length()

    def _update_length(self):
        if len(self.data) == 0:
            self.length = 0
            return

        value_lengths = list(map(len, self.data.values()))
        assert len(set(value_lengths)) == 1, 'All tensors in the batch must have the same length'
        self.length = value_lengths[0]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def __getitem__(self, index):
        """Can index the data either using integers, slices or by dict keys.  Indexing by integer / slice returns a
        dictionary, indexing by dict keys (strings) returns the corresponding numpy array. """
        if isinstance(index, int) or isinstance(index, slice):
            return Batch({key: value[index] for key, value in self.data.items()})
        elif isinstance(index, str):
            return self.data[index]
        else:
            raise TypeError(f'Invalid type: {type(index)}')

    def __iadd__(self, batch):
        if self.data == {}:
            self.data = batch.data.copy()
        else:
            err_msg = f'{set(self.data.keys())} mismatch {set(batch.data.keys())}'
            assert self.data.keys() == batch.data.keys(), err_msg
            for key, value in batch.data.items():
                self.data[key] = np.concatenate((self.data[key], value))
        self._update_length()
        return self

    def __len__(self):
        return self.length

    def __str__(self):
        return f'{self.__class__.__name__}({self.data})'


class InMemoryDataset(Dataset):
    """Represents a dataset that can be loaded into memory."""
    def __init__(self, data_files: List[Path], data_file_lengths: List[int], start_index: int, length: int,
                 dtype: type):
        assert len(data_files) > 0
        self.dtype = dtype
        self.length = length

        self.inputs = self._load_data(data_files, data_file_lengths, start_index)

    def _get_slice(self, array: np.ndarray, start_index: int, end_index: int) -> np.ndarray:
        if start_index < 0:
            start_index = 0
        if end_index > len(array):
            end_index = len(array)
        return array[start_index:end_index]

    def _load_data(self, data_files: List[Path], data_file_lengths: List[int], start_index: int):
        """Load data starting from start_index into the dataset.  Stop loading data once the dataset size becomes
        equal to the length defined in the __init__ constructor."""
        end_index = start_index + self.length
        inputs = defaultdict(list)
        for file_, file_length in zip(data_files, data_file_lengths):
            # start loading input data into memory
            if start_index < file_length:
                self._load_data_single_file(inputs, file_, start_index, end_index)
            # stop loading input data into memory
            if end_index < file_length:
                break
            start_index -= file_length
            end_index -= file_length
        # concatenate the numpy arrays of input data across files
        return {key: np.concatenate(tuple(arrays)) for key, arrays in inputs.items()}

    def _load_data_single_file(self, inputs: np.ndarray, hdf5_file: h5py.File, start_index: int, end_index: int):
        with h5py.File(hdf5_file) as f:
            for key, value in f.items():
                input_data = self._get_slice(value, start_index, end_index)
                inputs[key].append(np.asarray(input_data))

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return {key: torch.from_numpy(input_[index].astype(self.dtype)) for key, input_ in self.inputs.items()}

    def __len__(self):
        return self.length
