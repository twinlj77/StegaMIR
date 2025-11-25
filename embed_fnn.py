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

"""Helper functions with message embedding."""
from typing import List, Optional, Tuple, Union

from absl import logging
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from functa import function_reps
from functa import pytree_conversions
from functa.minimal_nerf import render_rays

from jax import random

Array = jnp.ndarray
PRNGKey = chex.PRNGKey

# Helper functions to compute MSE and PSNR
mse_fn = jax.jit(lambda x, y: jnp.mean((x - y) ** 2))
psnr_fn = jax.jit(lambda mse: -10 * jnp.log10(mse))
inverse_psnr_fn = jax.jit(lambda psnr: jnp.exp(-psnr * jnp.log(10) / 10))


#定义随机生成秘密消息original_msg
def generate_bit_string(key, length=50):
    """生成一个长度为 length 的 0-1 比特串"""
    normal_random = random.normal(key, shape=(length,))  # 使用正态分布生成随机数
    bits = (normal_random > 0).astype(jnp.float32)  # 将大于 0 的值设为 1，小于等于 0 的值设为 0
    return bits


key = random.PRNGKey(17)  #设置随机数种子
bit_string = generate_bit_string(key, length=50)  #生成30维的 0-1 比特串
original_msg = 2 * bit_string - 1  # 将 0/1 映射到 -1/1


#original_msg = apply_softmax(bit_string)#应用softmax 函数
# 打印结果
#print("\n原秘密消息:", original_msg)

#定义固定消息提取器
# 定义可微但固定的 extractor 网络
class MLPExtractor(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x):
        x = x.reshape(-1)
        residual = x
        x = hk.Linear(4096)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(4096)(x)
        x = jax.nn.relu(x + residual[:4096])
        x = hk.Linear(50)(x)
        x = jax.nn.tanh(x)
        return x


# 使用 hk.transform 包装 extractor
extractor_fn = hk.without_apply_rng(hk.transform(lambda x: MLPExtractor()(x)))

# 初始化 extractor 参数
dummy_input = jnp.zeros((256 * 256 * 3,))
fixed_extractor_params = extractor_fn.init(random.PRNGKey(17), dummy_input)


# 定义特征适配器网络
class FeatureAdapter(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.conv1 = hk.Conv2D(256, kernel_shape=3, stride=1, padding="SAME")
        self.conv2 = hk.Conv2D(256, kernel_shape=3, stride=1, padding="SAME")

    def __call__(self, x):
        # 添加批次维度 (1, H, W, C)
        x = x[jnp.newaxis, ...]
        x = self.conv1(x)
        x = jax.nn.leaky_relu(x)
        x = self.conv2(x)
        x = jax.nn.leaky_relu(x)  # 输出范围 [-1, 1]
        return x[0]  # 移除批次维度


# 创建适配器转换函数
adapter_fn = hk.without_apply_rng(hk.transform(lambda x: FeatureAdapter()(x)))

# 初始化适配器参数 (使用图像形状的虚拟输入)
dummy_img = jnp.zeros((256, 256, 3))  # 假设图像大小为 64x64
adapter_params = adapter_fn.init(random.PRNGKey(17), dummy_img) #原42


#计算消息提取准确率
def calculate_accuracy(extracted_msg: Array, original_msg: Array) -> Array:
    """计算提取消息的比特准确率"""
    predicted_bits = jnp.sign(extracted_msg)  # 将连续值转换为二进制比特（-1/1）
    # 计算准确率
    accuracy = jnp.mean(jnp.equal(predicted_bits, original_msg))
    return accuracy


def compute_grad_norm(grads):
    """计算梯度的L2范数"""
    grads_flat, _ = jax.tree_util.tree_flatten(grads)
    return jnp.sqrt(sum(jnp.sum(g ** 2) for g in grads_flat))


def loss_fn_image(modulations: hk.Params, weights: hk.Params, model,
                  image: Array, coords: Array, l2_weight: float,
                  original_msg: Array, msg_weight: float = 0.5, adapter_params: hk.Params = None,
                  return_msg_loss: bool = False) -> Tuple[Array, Tuple]:  #Array:
    """Loss function for images.

  Args:
    modulations: Modulation parameters.
    weights: Shared weight parameters.
    model: Haiku transformed model.
    image: Shape (height, width, channels).
    coords: Shape (height, width, 2) or (height * width, 2). Note the coords
      will be flattened in model call.
    l2_weight: weight for L2 regularisation of modulations.

  Returns:
    MSE between ground truth image and image reconstructed by function rep.
  """
    params = function_reps.merge_params(weights, modulations)
    generated = model.apply(params, coords)
    modulations_array, _, _ = pytree_conversions.pytree_to_array(modulations)
    l2_loss = l2_weight * jnp.sum(modulations_array ** 2)
    rec_loss = mse_fn(generated, image)
    #return rec_loss + l2_loss, rec_loss 原

    # 应用特征适配器
    if adapter_params:
        adapted_img = adapter_fn.apply(adapter_params, generated)
    else:
        adapted_img = generated

    extracted_msg = extractor_fn.apply(fixed_extractor_params, generated.reshape(-1))
    accuracy = calculate_accuracy(extracted_msg, original_msg)
    #jax.debug.print("提取的秘密消息: {}", extracted_msg) #运行时输出实际值

    # 增强的消息损失函数
    sign_loss = jnp.mean(jnp.maximum(0, 1.0 - extracted_msg * original_msg))
    weighted_mse = jnp.mean((extracted_msg - original_msg) ** 2)
    msg_loss = 0.7 * sign_loss + 0.3 * weighted_mse
    total_loss = rec_loss + l2_loss + msg_weight * msg_loss

    #jax.debug.print("Total Loss = {}, Rec Loss = {}, Acc = {}",
    #total_loss, rec_loss, accuracy)
    #jax.debug.print("Total Loss = {}, Rec Loss = {}, Msg Loss = {}, Acc = {}",
    #total_loss, rec_loss, msg_loss, accuracy)
    if return_msg_loss:
        # 只返回消息损失部分
        return msg_loss
    else:
        total_loss = rec_loss + l2_loss + msg_weight * msg_loss
        return total_loss, (rec_loss, accuracy, msg_loss)  # 返回总损失和重建损失


def loss_fn_nerf(modulations: hk.Params, weights: hk.Params, model,
                 target: Array, rays: Array,
                 render_config: Tuple[int, float, float, bool],
                 l2_weight: float, rng: Union[int, PRNGKey] = 17,
                 coord_noise: bool = False):
    """Loss function for scenes.

  Args:
    modulations: Modulation parameters.
    weights: Shared weight parameters.
    model: Haiku transformed model.
    target: Target pixel values for a single or a batch of images
      *of the same scene*. Shape (H, W, 3) or (num_views, H, W, 3).
    rays: Ray origin and direction for each target value.
      Shape (2, H, W, 3) or (2, num_views, H, W, 3).
    render_config: config for nerf.
    l2_weight: weight for L2 regularisation of modulations.
    rng: PRNG key for adding coordinate noise.
    coord_noise: whether to add coordinate noise or not.

  Returns:
    loss: scalar MSE between ground truth view and image reconstructed by
      function rep.
  """
    params = function_reps.merge_params(weights, modulations)
    rgb, _ = render_rays(model, params, rays, render_config, rng, coord_noise)
    modulations_array, _, _ = pytree_conversions.pytree_to_array(modulations)
    l2_loss = l2_weight * jnp.sum(modulations_array ** 2)
    rec_loss = mse_fn(rgb, target)
    return rec_loss + l2_loss, rec_loss


def inner_loop(
        params: hk.Params,
        model,
        opt_inner: optax.GradientTransformation,
        inner_steps: int,
        coords: Array,
        targets: Array,
        return_all_psnrs: bool = False,
        return_all_losses: bool = False,
        is_nerf: bool = False,
        render_config: Optional[Tuple[int, float, float, bool]] = None,
        l2_weight: float = 0.,
        noise_std: float = 0.,
        rng: Union[int, PRNGKey] = 17,
        coord_noise: bool = False,
        original_msg: Array = None,  # 新增参数
        msg_weight: float = 0.5,
        return_extracted_msg: bool = False,  # 新增参数
        adapter_params: hk.Params = None,  # 新增适配器参数
) -> Union[Tuple[hk.Params, Array, Array], Tuple[
    hk.Params, Array, Array, List[Array]], Tuple[hk.Params, Array, List[Array]],
Tuple[hk.Params, Array, List[Array], List[Array]]]:
    """Performs MAML (Finn et al.'17) inner loop: fits modulations to target data.

  This function takes `inner_steps` SGD steps in the inner loop to fit
  modulations to image, while keeping weights fixed. This function is applied
  to a single target (e.g. image, video or 3d scene).

  Args:
    params: ModulatedSiren model params.
    model: Haiku transformed model.
    opt_inner: Optax optimizer (typically SGD).
    inner_steps: Number of SGD steps to take to fit modulations to image.
    coords: Coordinates at which function rep will be evaluated.
    targets: Data to be fitted. Not batched. For example, a single image of
      shape (height, width, 3).
    return_all_psnrs: If True, returns a list of PSNRs at every step during
      fitting, otherwise returns only final PSNR.
    return_all_losses: If True, returns a list of losses at every step during
      fitting. Only comes into effect when return_all_psnrs=True.
    is_nerf: If True, uses nerf inner loop.
    render_config: config for nerf.
    l2_weight: weight for L2 regularisation of modulations.
    noise_std: standard deviation of Gaussian noise applied to modulations.
    rng:
    coord_noise: whether to add coordinate noise or not. Only used if
      `is_nerf=True`.

  Returns:
    Fitted params, loss and either final PSNR or all PSNR values.
  """

    if isinstance(rng, int):
        rng = jax.random.PRNGKey(rng)
    # Partition params into trainable modulations and non-trainable weights
    weights, modulations = function_reps.partition_params(params)

    # Check if 'meta_sgd_lrs' is inside a key in weights. If it is, use meta-SGD
    # to fit the data
    use_meta_sgd = False
    for key in weights:
        if 'meta_sgd_lrs' in key:
            use_meta_sgd = True

            # Detailed step metrics
    step_metrics = {
        'rec_losses': [],
        'psnrs': [],
        'accuracies': [],
        'grad_norms': [],
        'msg_losses': []
    }

    # Initialize metrics storage
    if return_all_psnrs:
        psnr_vals = []
    if return_all_losses:
        loss_vals = []

    if use_meta_sgd:
        # Extract learning rates
        _, lrs = function_reps.partition_shared_params(weights)
        # Flatten lrs so they can easily be multiplied with modulations when
        # performing meta-SGD update
        flat_lrs, _, _ = pytree_conversions.pytree_to_array(lrs)

    # Inner optimizer should have no memory of its state, every time we do inner
    # loop optimization we are solving a new problem from scratch, so optimizer
    # should be reinitialized. As we only update modulations with opt_inner,
    # initialize with modulations and not all params
    # Only use optimizer if we are not using meta-SGD (where we learn learning
    # rates per parameter)
    if not use_meta_sgd:
        opt_inner_state = opt_inner.init(modulations)

    # Only update modulations in inner loop
    for step in range(inner_steps):
        # jax.grad takes gradient with respect to first positional argument only
        if is_nerf:
            (loss, rec_loss), modulations_grad = jax.value_and_grad(
                loss_fn_nerf, has_aux=True)(modulations, weights, model, targets,
                                            coords, render_config, l2_weight,
                                            rng, coord_noise)
        else:
            (loss, (rec_loss, accuracy, msg_loss)), modulations_grad = jax.value_and_grad(
                loss_fn_image, has_aux=True)(modulations, weights, model, targets,
                                             coords, l2_weight, original_msg, msg_weight, adapter_params)

        # 计算梯度范数
        grad_norm = compute_grad_norm(modulations_grad)

        # 记录指标
        step_psnr = psnr_fn(rec_loss)
        step_metrics['rec_losses'].append(rec_loss)
        step_metrics['psnrs'].append(step_psnr)
        step_metrics['accuracies'].append(accuracy)
        step_metrics['grad_norms'].append(grad_norm)
        step_metrics['msg_losses'].append(msg_loss)

        # 打印每一步的指标
        print(f"  Inner Step {step + 1}/{inner_steps}: " f"PSNR: {step_psnr:.2f} dB, "
              f"Accuracy: {accuracy * 100:.2f}%, " f"Grad Norm: {grad_norm:.6f}, "
              f"Rec Loss: {rec_loss:.6f}, " f"Msg Loss: {msg_loss:.6f}")

        # Update modulations
        if use_meta_sgd:
            # modulations_grad is a pytree with the same keys as modulations. lrs is
            # a pytree containing all learning rates as a single array in a single
            # leaf. Flatten both to multiply them together and then reconstruct tree
            # Note, learning rate flattening operation is done above, and we therefore
            # apply flat_lrs here
            # Note, the following two lines are awkward, but are required to satisfy
            # linter (line-too-long).
            out = pytree_conversions.pytree_to_array(modulations_grad)
            flat_modulations_grads, concat_idx, tree_def = out
            flat_modulations_updates = -flat_lrs * flat_modulations_grads
            modulation_updates = pytree_conversions.array_to_pytree(
                flat_modulations_updates, concat_idx, tree_def)
        else:
            modulation_updates, opt_inner_state = opt_inner.update(
                modulations_grad, opt_inner_state)
        # Apply gradient update
        modulations = optax.apply_updates(modulations, modulation_updates)

        # Optionally calculate PSNR value
        if return_all_psnrs:
            psnr_vals.append(psnr_fn(rec_loss))
        if return_all_losses:
            loss_vals.append(loss)

    # Optionally add noise to fitted modulations, to make downstream task less
    # sensitive to exact value of modulations.
    if noise_std > 0.:
        modulations_array, concat_idx, tree_def = pytree_conversions.pytree_to_array(
            modulations)
        modulations_array += noise_std * jax.random.normal(
            rng, shape=modulations_array.shape)
        modulations = pytree_conversions.array_to_pytree(modulations_array,
                                                         concat_idx, tree_def)

    # Compute final loss using updated modulations
    if is_nerf:
        loss, rec_loss = loss_fn_nerf(modulations, weights, model, targets, coords,
                                      render_config, l2_weight, rng, coord_noise)
    else:
        loss, (rec_loss, accuracy, msg_loss) = loss_fn_image(modulations, weights, model, targets, coords,
                                                             l2_weight, original_msg, msg_weight)

    total_loss = loss

    # Add final metrics to step_metrics
    final_psnr = psnr_fn(rec_loss)
    step_metrics['rec_losses'].append(rec_loss)
    step_metrics['psnrs'].append(psnr_fn(rec_loss))
    step_metrics['accuracies'].append(accuracy)
    step_metrics['msg_losses'].append(msg_loss)

    # Compute final gradient norm (for completeness)
    if is_nerf:
        (_, _), modulations_grad = jax.value_and_grad(
            loss_fn_nerf, has_aux=True)(modulations, weights, model, targets,
                                        coords, render_config, l2_weight,
                                        rng, coord_noise)
    else:
        (_, (_, _, _)), modulations_grad = jax.value_and_grad(
            loss_fn_image, has_aux=True)(modulations, weights, model, targets,
                                         coords, l2_weight, original_msg, msg_weight)
    final_grad_norm = compute_grad_norm(modulations_grad)
    step_metrics['grad_norms'].append(final_grad_norm)

    # Print final metrics
    print(f"  Final: PSNR: {final_psnr:.2f} dB, " f"Accuracy: {accuracy * 100:.2f}%, "
          f"Grad Norm: {final_grad_norm:.6f}, " f"Rec Loss: {rec_loss:.6f}, "
          f"Msg Loss: {msg_loss:.6f}")

    # 返回参数，总损失，psnr值，提取的消息，准确度  if return_all_psnrs:
    #    psnr_vals.append(psnr_fn(rec_loss))

    #  if return_all_losses:
    #    loss_vals.append(loss)

    # Merge weights and modulations and return
    params = function_reps.merge_params(weights, modulations)

    #  if return_extracted_msg:
    # 生成最终图像并提取消息
    #      params_final = function_reps.merge_params(weights, modulations)
    #      generated = model.apply(params_final, coords)
    #      extracted_msg = extractor_fn.apply(fixed_extractor_params, generated(-1))
    #      if return_all_psnrs and not return_all_losses:
    #          return params, total_loss, psnr_vals, extracted_msg, accuracy
    #      elif return_all_psnrs and return_all_losses:
    #          return params, total_loss, psnr_vals, loss_vals, extracted_msg, accuracy
    #      else:
    #          return params, total_loss, psnr_fn(rec_loss), extracted_msg, accuracy #这一大段新加
    #  else:
    #      if return_all_psnrs and not return_all_losses:
    #         return params, total_loss, psnr_vals, step_metrics
    #      elif return_all_psnrs and return_all_losses:
    #         return params, total_loss, psnr_vals, loss_vals, step_metrics
    #      elif not return_all_psnrs and return_all_losses:
    #          return params, total_loss, loss_vals, step_metrics
    #      else:
    #         return params, total_loss, step_metrics

    # Prepare return values
    return_values = [params, total_loss]

    if return_all_psnrs:
        return_values.append(psnr_vals)
    if return_all_losses:
        return_values.append(loss_vals)

    # Always return step metrics
    return_values.append(step_metrics)

    return tuple(return_values)


def image_grid_from_batch(images: Array) -> Array:
    """Simple helper to generate a single image from a mini batch.

  Args:
    images: Batch of images of shape (batch_size, height, width, channels)

  Returns:
    A single image of shape (img_grid_height, img_grid_width, channels).
  """
    batch_size = images.shape[0]
    grid_size = int(np.floor(np.sqrt(batch_size)))

    img_iter = iter(images[0:grid_size ** 2])
    return jnp.squeeze(
        jnp.vstack([
            jnp.hstack([next(img_iter)
                        for _ in range(grid_size)][::-1])
            for _ in range(grid_size)
        ]))


def log_params_info(params):
    """Log information about parameters."""
    logging.info('Parameter shapes')
    logging.info(jax.tree_map(jnp.shape, params))
    num_params = hk.data_structures.tree_size(params)
    byte_size = hk.data_structures.tree_bytes(params)
    logging.info('%d params, size: %.2f MB', num_params, byte_size / 1e6)
    # print each parameter and its shape
    for mod, name, value in hk.data_structures.traverse(params):
        logging.info('%s/%s: %s', mod, name, value.shape)
