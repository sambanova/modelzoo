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

import csv
import os
from typing import Any, Dict, List


def save_summary_report(output_dir: str, filename: str, summary_text: str) -> None:
    """
    Save a training summary report

    Args:
        output_dir: directory to store report
        summary_text: text to output in summary file
    """
    output_file = os.path.join(output_dir, filename)
    with open(output_file, 'w') as out_file:
        out_file.write(summary_text)


def save_per_step_report(output_dir: str, filename: str, metrics: Dict[str, List[Any]]) -> None:
    """
    Save a csv report of per step metrics
    
    Args:
        output_dir: directory to store report
        metrics: dict mapping metric name to list of metrics per step
    """
    output_file = os.path.join(output_dir, filename)
    with open(output_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
        writer.writeheader()
        for row in zip(*metrics.values()):
            row_dict = {key: value for key, value in zip(metrics.keys(), row)}
            writer.writerow(row_dict)
