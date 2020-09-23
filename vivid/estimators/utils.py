from tabulate import tabulate
from typing import List


def to_pretty_lines(input_dict: dict) -> List[str]:
    s_metric = tabulate([input_dict], headers='keys', tablefmt='github')
    lines = [l for l in s_metric.split('\n')]
    return lines
