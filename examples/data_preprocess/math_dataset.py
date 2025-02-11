# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'mlfoundations-dev/math_stratos_scale'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    # First pass: Filter valid examples
    def validate_example(example):
        try:
            answer = example['solution']
            solution = extract_solution(answer)
            return True
        except Exception as e:
            return False

    # Get initial stats
    total_examples = len(dataset['train'])
    print(f"Initial dataset size: {total_examples}")

    # Filter valid examples
    filtered_dataset = dataset['train'].filter(validate_example)
    valid_examples = len(filtered_dataset)
    print(f"Valid examples after filtering: {valid_examples}")
    print(f"Removed {total_examples - valid_examples} examples")

    # Second pass: Split into train/test
    split_dataset = filtered_dataset.train_test_split(
        test_size=1000, 
        shuffle=True, 
        seed=42
    )
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    print(f"Final train set size: {len(train_dataset)}")
    print(f"Final test set size: {len(test_dataset)}")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # Continue with the existing processing...
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop('problem')
            question = question + ' ' + instruction_following

            answer = example.pop('solution')
            solution = extract_solution(answer)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn

    # Process the datasets
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
