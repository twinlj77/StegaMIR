# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper functions to convert between pytree and array representations."""

import jax
import jax.numpy as jnp


def flattened_pytree_to_array(flattened_pytree):
  """Converts a flattened pytree to a single concatenated array.

  Args:
    flattened_pytree (List of Array): List of arrays returned from
      jax.tree_flatten. Note each array must be 1-dimensional.

  Returns:
    Concatenated array and concatenation indices.
  """
  # Extract concatenation indices so we can later "unconcatenate" array to
  # recreate pytree
  concat_idx = []   #初始化一个空列表concat_idx，用于存储每个展平数组的长度。
  current_idx = 0   #初始化一个变量current_idx，用于记录当前索引的位置，初始值为0。
  for np_array in flattened_pytree:
    current_idx += len(np_array) #计算当前np_array的长度，并将其加到current_idx上
    concat_idx.append(current_idx) #将更新后的current_idx值添加到concat_idx列表中
  # Return concatenated pytree and concatenation indices
  return jnp.concatenate(flattened_pytree), concat_idx #使用jnp.concatenate函数将flattened_pytree中的所有一维数组合并成一个大的一维数组;
  # 返回这个合并后的数组以及之前计算的concat_idx列表


def array_to_flattened_pytree(concat_array, concat_idx):
  """Converts a concatenated numpy array to a list of numpy arrays.

  Args:
    concat_array (Array):
    concat_idx (List of int):

  Returns:
    A flattened pytree (i.e. a list of numpy arrays).

  Notes:
    Inverse function of flattened_pytree_to_array.
  """
  # Split array according to concat idx
  flattened_pytree = []
  prev_idx = 0
  for idx in concat_idx:
    flattened_pytree.append(concat_array[prev_idx:idx])
    prev_idx = idx
  return flattened_pytree


def pytree_to_array(pytree):
  """Converts a pytree to single concatened array.

  Args:
    pytree (Pytree):

  Returns:
    Concatenated array, concatenation indices and tree definition which are
    required to reconstruct pytree.

  Notes:
    Note that pytree must contain only one dimensional tensors (as is the case
    for example with a pytree of modulations).
  """
  flattened_pytree, tree_def = jax.tree_util.tree_flatten(pytree)
  concat_array, concat_idx = flattened_pytree_to_array(flattened_pytree)
  return concat_array, concat_idx, tree_def


def array_to_pytree(concat_array, concat_idx, tree_def):
  """Converts a concatenated array to a pytree.

  Args:
    concat_array (Array):
    concat_idx (List of int):
    tree_def (TreeDef):

  Returns:
    The reconstructed pytree.
  """
  flattened_pytree = array_to_flattened_pytree(concat_array, concat_idx)
  return jax.tree_util.tree_unflatten(tree_def, flattened_pytree)
