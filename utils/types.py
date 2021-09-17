from typing import Tuple, List


def group_list(input_list: List, n: int) -> List[Tuple]:
    """
    Create groups of n within list.
    :param input_list: input list.
    :param n: group size.
    :return: grouped list.
    """
    return list(zip(*(iter([int(v) for v in input_list]),) * n))


def is_number(s: str) -> bool:
    """
    Whether a string is a number (including negative and scientific notation).
    :param s: string.
    :return: true or false.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def flatten_list(input_list: List) -> List:
    """
    Flatten a list of lists.
    :param input_list: List of lists.
    :return: Flat list.
    """
    return [item for sublist in input_list for item in sublist]
