# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.collection_utils import as_iter, common_prefix, swap_dict_values


@pytest.mark.parametrize(
    ("inputs", "expected_output"),
    [
        ([""], ""),
        (["a"], "a"),
        (["a", "b"], ""),
        (["a", "ab"], "a"),
        (["a", "ab", "b"], ""),
        (["abc", "a", "ab"], "a"),
        (["aba", "abc", "ab"], "ab"),
    ],
)
def test_common_prefix(inputs, expected_output):
    assert common_prefix(inputs) == expected_output


@pytest.mark.parametrize(
    ("obj", "key1", "key2"),
    [
        # Tests for both keys exist
        ({1: "a", 2: "b"}, 1, 2),
        # Tests for one key does not exist
        ({1: "a", 2: "b"}, 1, 3),
        # Tests for both keys do not exist
        ({1: "a", 2: "b"}, 3, 4),
    ],
)
def test_swap_dict_values(obj, key1, key2):
    original_obj = obj.copy()

    swap_dict_values(obj, key1, key2)

    if key1 in original_obj:
        assert obj[key2] == original_obj[key1]
    else:
        assert key2 not in obj
    if key2 in original_obj:
        assert obj[key1] == original_obj[key2]
    else:
        assert key1 not in obj


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("abc", ["abc"]),
        (7, [7]),
        ([1, 2], [1, 2]),
        ((1, 2), (1, 2)),
    ],
)
def test_as_iter_wraps_scalars_but_keeps_iterables(value, expected):
    assert list(as_iter(value)) == list(expected)
