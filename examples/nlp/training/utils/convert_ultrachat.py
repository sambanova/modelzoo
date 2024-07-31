# Copyright 2024 SambaNova Systems, Inc.
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

import argparse
import io
import json
import os


def process_ultrachat_jsonl(src_file: str, dest_handle: io.TextIOWrapper) -> None:
    """
    Transform one UltraChat file to prompt and completion pairs

    Arguments:
    - src_file: Full path of .jsonl file to read
    - dest_handle: Open file handle to output .jsonl file

    Side effects:
    - Writes list of prompt-completion pairs to dest_handle
        * Each line in output file is a list of dicts:
          [{"prompt": "", "completion": ""}, ...]
        * Each line represents one full conversation
        * Each prompt-completion pair represents a
          human question (prompt) and bot answer (completion)
    """

    # There are a couple of json errors in the dataset
    # We'll keep track and let the user know
    skipped_samples = 0

    with open(src_file, 'r') as f:

        print(f'Working on file:', src_file)
        for line in f:  # new conversation in ultrachat
            conversation = []
            try:
                sentences = json.loads(line)['data']  # Turn by turn human then bot
            except json.JSONDecodeError as e:
                # Skip this line but let the user know
                skipped_samples += 1
                continue
            current_pair = {}
            for turn, sentence in enumerate(sentences):
                if turn % 2:  # Assistant
                    current_pair["completion"] = sentence
                    conversation.append(current_pair)
                else:  # Human
                    current_pair["prompt"] = sentence
            json.dump(conversation, dest_handle)
            dest_handle.write('\n')

    if skipped_samples:
        print(f'Skipped {skipped_samples} {"lines" if skipped_samples > 1 else "line"} due to errors')


def main(src, dest):
    """
    Process UltraChat dataset to format expected by Generative Data Prep

    Arguments:
    - src: Relative path to the UltraChat dataset folder
    - dest: Relative path to output .jsonl file for Generative Data Prep
    """

    abspath = os.path.abspath(src)
    src_files = filter(lambda x: x.endswith('.jsonl'), os.listdir(abspath))

    with open(os.path.abspath(dest), 'w') as outfile:
        for f in sorted(src_files):
            process_ultrachat_jsonl(f'{abspath}/{f}', outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', required=True)
    parser.add_argument('-dest', default='ultrachat_processed.jsonl')

    args = parser.parse_args()

    main(args.src, args.dest)
