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

"""Create modulation dataset with embedded messages."""

import os

import jax
from absl import app
from absl import flags
import dill
import haiku as hk
import numpy as np
import optax

import save
from functa import data_utils
from functa import function_reps
from functa import embed_fnn
from functa import pytree_conversions


flags.DEFINE_float('msg_lr', 0.5, 'Learning rate for message embedding optimization')
flags.DEFINE_float('msg_weight', 0.05, 'Weight for message loss')
flags.DEFINE_integer('total_steps', 100, 'Total inner loop steps (recon + msg)')
FLAGS = flags.FLAGS

flags.DEFINE_integer('mod_dim', 64,
                     'The dimensionality of modulation dimension to use.'
                     'Choose one of: 64, 128, 256, 512, 1024.')
flags.DEFINE_string('pretrained_weights_dir', './',
                    'Path to directory containing pre-trained weights.')
flags.DEFINE_string('save_to_dir', 'save/embed/bits/128',
                    'Path to directory where modulations should be saved.')


def create_modulation_dataset(model, params, ds, num_steps, coords, lr,
                              l2_weight, noise_std):
    """Creates dataset of modulations with embedded messages."""
    # 初始化消息相关组件
    original_msg = embed_fnn.original_msg
    extractor_params = embed_fnn.fixed_extractor_params

    # 组合优化器：前3步用SGD优化重建，后续用Adam优化消息
    opt_recon = optax.sgd(lr)
    opt_msg = optax.adam(FLAGS.msg_lr)

    mod_list = []
    psnr_list = []
    rec_loss_list = []  # 重建损失记录
    msg_acc_list = []  # 消息准确率记录

    for i, datum in enumerate(ds):

        fitted_params, total_loss, psnr_vals, loss_vals, step_metrics = embed_fnn.inner_loop(
            params=params,
            model=model,
            opt_inner=opt_recon,
            inner_steps=FLAGS.total_steps,
            coords=coords,
            targets=datum['array'],
            return_all_psnrs=True,
            return_all_losses=True,
            l2_weight=l2_weight,
            noise_std=noise_std,
            original_msg=original_msg,
            msg_weight=FLAGS.msg_weight
        )
    # for i, datum in enumerate(ds):
        # 计算最终重建损失
        weights, modulations = function_reps.partition_params(fitted_params)
        if hasattr(embed_fnn, 'loss_fn_image') and embed_fnn.loss_fn_image is not None:
            #_, rec_loss = embed_fnn.loss_fn_image(
            #    modulations, weights, model, datum['array'], coords, l2_weight)
            result = embed_fnn.loss_fn_image(
                modulations, weights, model, datum['array'], coords, l2_weight,
                original_msg, FLAGS.msg_weight)
            rec_loss = result[1][0]  # 获取重建损失
        else:
            rec_loss = loss_vals[-1]

        # 计算消息准确率
        generated_img = model.apply(fitted_params, coords)
        extracted_msg = embed_fnn.extractor_fn.apply(embed_fnn.fixed_extractor_params, generated_img)
        msg_acc = embed_fnn.calculate_accuracy(extracted_msg, original_msg)

        # 收集数据
        mod_array, _, _ = pytree_conversions.pytree_to_array(modulations)
        mod_list.append(mod_array)
        psnr_list.append(psnr_vals[-1])
        rec_loss_list.append(rec_loss)
        msg_acc_list.append(msg_acc)

        #print(f"Step {i + 1}: Avg Msg Acc: {np.mean(msg_acc_list):.2%}")
        print(f'data point {i + 1} has: PSNR {psnr_vals[-1]:.2f}dB | Msg Acc {msg_acc * 100:.2f}%')
    return {
        'modulations': np.stack(mod_list),
        'psnr': np.array(psnr_list),
        'rec_loss': np.array(rec_loss_list),
        'msg_accuracy': np.array(msg_acc_list),
        'original_msg': original_msg,
        'extractor_params': extractor_params
    }


def main(_):

    mod_dim = FLAGS.mod_dim
    assert mod_dim in [64, 128, 256, 512, 1024], f'Invalid mod_dim: {mod_dim}'

    path = os.path.join(FLAGS.pretrained_weights_dir,
                        f'celeba_params_{mod_dim}_latents.npz')
    assert os.path.exists(path), 'Pretrained weights file does not exist.'

    with open(path, 'rb') as f:
        ckpt = dill.load(f)
    params = ckpt['params']
    config = ckpt['config']

    # 构建模型（保持原有逻辑）
    model_config = config['model'].copy()
    model_config.pop('type', None)
    model_config.pop('l2_weight', None)
    model_config.pop('noise_std', None)

    def model_net(coords):
        hk_model = function_reps.LatentModulatedSiren(
            out_channels=config['dataset']['num_channels'], **model_config)
        return hk_model(coords)

    model = hk.without_apply_rng(hk.transform(model_net))

    # 数据集准备
    train_ds = data_utils.load_dataset('celeb_a_hq256', subset='train',num_examples=30)
    #test_ds = data_utils.load_dataset('celeb_a_hq_custom', subset='test')
    coords = function_reps.get_coordinate_grid(config['dataset']['resolution'])

    # 生成训练集调制参数
    print("\nGenerating training modulations with messages...")
    train_data = create_modulation_dataset(
        model=model,
        params=params,
        ds=train_ds,
        num_steps=config['training']['inner_steps'],
        coords=coords,
        lr=config['opt_inner']['lr'],
        l2_weight=config['model']['l2_weight'],
        noise_std=config['model']['noise_std'])

    # 生成测试集调制参数
    #print("\nGenerating test modulations with messages...")
    #test_data = create_modulation_dataset(
    #model=model,
    #params=params,
    #ds=test_ds,
    #num_steps=config['training']['inner_steps'],
    #coords=coords,
    #lr=config['opt_inner']['lr'],
    #l2_weight=config['model']['l2_weight'],
    #noise_std=config['model']['noise_std'])

    # 保存完整数据集
    modulation_data = {
        'train': {
            'modulations': train_data['modulations'],
            'psnr': train_data['psnr'],
            'rec_loss': train_data['rec_loss'],
            'msg_accuracy': train_data['msg_accuracy']
        },
        #'test': {
        #   'modulations': test_data['modulations'],
        #  'psnr': test_data['psnr'],
        #   'rec_loss': test_data['rec_loss'],
        #   'msg_accuracy': test_data['msg_accuracy']
        #},
        'meta': {
            'original_msg': train_data['original_msg'],
            'extractor_params': train_data['extractor_params'],
            'config': {
                'msg_lr': FLAGS.msg_lr,
                #'msg_steps': FLAGS.msg_steps,
                'total_steps': FLAGS.total_steps,
                'msg_weight': FLAGS.msg_weight
            }
        }
    }

    save_path = os.path.join(FLAGS.save_to_dir,
                             f'celeba_modulations_{mod_dim}_latents_32bits.npz')
    with open(save_path, 'wb') as f:
        dill.dump(modulation_data, f)
    print(f"\nSaved embedded modulations to {save_path}")
    #path = "celeba_modulations_512_latents_512.npz"
    #save.runsave(save_path)

if __name__ == '__main__':
    app.run(main)

