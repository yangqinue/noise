# Copyright (c) Xuechen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper launcher script."""

import os

import fire

from .src import common


def _get_command(
    task_name,
    output_dir,
    noise_multiplier,
    model_name_or_path,
    data_dir,
    learning_rate,
    clipping_mode: str,
    non_private,
    target_epsilon,
    few_shot_type,
    seed,
    attention_only,
    static_lm_head,
    static_embedding,
    randomly_initialize,
    per_device_train_batch_size,
    batch_size,
    num_train_epochs,
    eval_steps,
    eval_spectrum,
    max_spectrum_batches,
    max_lanczos_iter,
    store_grads,
    per_example_max_grad_norm,
    lmo_filepath,
    accounting_mode,
    orthogonal_projection_path,
    orthogonal_projection_rank,
):
    task_name_to_factor = {
        "sst-2": 1, "qnli": 2, "qqp": 6, "mnli": 6,
    }
    factor = task_name_to_factor[task_name]

    if batch_size is None:
        base_batch_size = 1000
        # This batch size selection roughly ensures the sampling rates on different
        # datasets are in the same ballpark.
        batch_size = int(base_batch_size * factor)
    gradient_accumulation_steps = batch_size // per_device_train_batch_size

    if num_train_epochs is None:
        base_num_train_epochs = 3
        num_train_epochs = int(base_num_train_epochs * factor)

    if learning_rate is None:
        if non_private.lower() in ('yes', 'y', 'true', 't'):
            learning_rate = 5e-5
        else:
            learning_rate = 5e-4

    data_dir = f"{data_dir}/{common.task_name2suffix_name[task_name]}"
    template = {
        "sst-2": "*cls**sent_0*_It_was*mask*.*sep+*",
        "mnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qnli": "*cls**sent-_0*?*mask*,*+sentl_1**sep+*",
        "qqp": "*cls**sent-_0**mask*,*+sentl_1**sep+*",
    }[task_name]

    # Epochs chosen roughly to match e2e number of updates. We didn't hyperparameter tune on classification tasks :)
    cmd = f'''
python3 -m classification.run_classification \
  --task_name {task_name} \
  --data_dir {data_dir} \
  --noise_multiplier {noise_multiplier} \
  --output_dir {output_dir} \
  --overwrite_output_dir \
  --model_name_or_path {model_name_or_path} \
  --few_shot_type {few_shot_type} \
  --num_k 1 \
  --num_sample 1 --seed {seed} \
  --template {template} \
  --non_private {non_private} \
  --num_train_epochs {num_train_epochs} \
  --target_epsilon {target_epsilon} --target_delta 1e-10 \
  --per_device_train_batch_size {per_device_train_batch_size} \
  --gradient_accumulation_steps {gradient_accumulation_steps} \
  --per_device_eval_batch_size 8 \
  --per_example_max_grad_norm {per_example_max_grad_norm} --clipping_mode {clipping_mode} \
  --learning_rate {learning_rate} \
  --lr_decay yes \
  --adam_epsilon 1e-08 \
  --weight_decay 0 \
  --max_seq_len 256 \
  --evaluation_strategy steps --eval_steps {eval_steps} --evaluate_before_training True \
  --do_train --do_eval --do_predict \
  --first_sent_limit 200 --other_sent_limit 200 --truncate_head yes \
  --attention_only {attention_only} --static_lm_head {static_lm_head} --static_embedding {static_embedding} \
  --randomly_initialize {randomly_initialize} \
  --eval_spectrum {eval_spectrum} --max_spectrum_batches {max_spectrum_batches} --max_lanczos_iter {max_lanczos_iter} \
  --store_grads {store_grads} \
  --accounting_mode {accounting_mode} 2>&1'''
    if orthogonal_projection_path is not None:
        cmd += f' --orthogonal_projection_path {orthogonal_projection_path}'
        cmd += f' --orthogonal_projection_rank {orthogonal_projection_rank}'
    return cmd


def main(
    output_dir,
    task_name,
    noise_multiplier,
    lmo_filepath=None,
    few_shot_type="prompt",
    seed=42,
    model_name_or_path="roberta-base",
    data_dir="classification/data/original",
    learning_rate=None,
    clipping_mode="ghost",
    non_private="no",
    target_epsilon=8,
    attention_only="no",
    static_lm_head="no",
    static_embedding="no",
    per_device_train_batch_size=20,
    eval_steps=10,
    eval_spectrum="no",
    max_spectrum_batches=2,
    max_lanczos_iter=2,
    randomly_initialize="no",
    batch_size=None,
    num_train_epochs=None,
    store_grads="no",
    per_example_max_grad_norm=1,
    accounting_mode='rdp',
    orthogonal_projection_path=None,
    orthogonal_projection_rank=100,
):
    command = _get_command(
        output_dir=output_dir,
        task_name=task_name,
        noise_multiplier=noise_multiplier,
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        learning_rate=learning_rate,
        clipping_mode=clipping_mode,
        non_private=non_private,
        target_epsilon=target_epsilon,
        few_shot_type=few_shot_type,
        seed=seed,
        attention_only=attention_only,
        static_lm_head=static_lm_head,
        static_embedding=static_embedding,
        per_device_train_batch_size=per_device_train_batch_size,
        eval_steps=eval_steps,
        eval_spectrum=eval_spectrum,
        max_spectrum_batches=max_spectrum_batches,
        max_lanczos_iter=max_lanczos_iter,
        randomly_initialize=randomly_initialize,
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        store_grads=store_grads,
        per_example_max_grad_norm=per_example_max_grad_norm,
        lmo_filepath=lmo_filepath,
        accounting_mode=accounting_mode,
        orthogonal_projection_path=orthogonal_projection_path,
        orthogonal_projection_rank=orthogonal_projection_rank,
    )
    print('Running command:')
    print(command)
    os.system(command)


if __name__ == "__main__":
    fire.Fire(main)
