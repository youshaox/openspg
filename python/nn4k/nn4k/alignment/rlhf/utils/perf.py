# coding: utf-8
import os

import torch
from transformers import AutoConfig

from alignment.util.global_vars import global_context


# This function can be used to print throughput for Step 1 and 2 only
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        run_ref_reward_async = global_context().model_conf.run_ref_reward_async

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
        if args.lora_dim > 0:
            k = args.lora_dim * 2 / hidden_size
            checkpoint_activations_factor -= (1 - k)

        hf_model._num_params = sum(
            [p.ds_numel if hasattr(p, "ds_tensor") else p.numel() for p in hf_model.parameters()])
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        train_flops_per_iteration = calculate_flops(checkpoint_activations_factor, batch_size, seq_length, hf_config)

        train_tflops = train_flops_per_iteration / (e2e_time * gpus_per_model * (10**12))

        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )


# Enhanced version of the function above that provides calculations and printing for Step 3
def print_throughput_step3(actor_model, critic_model, e2e_time, gen_exp_time, train_time, is_ac_share, rank=0):
    if rank <= 0:
        model_conf = global_context().model_conf
        train_conf = global_context().train_conf
        runtime_conf = global_context().runtime_conf

        # Actor model passed here is a HF model.

        actor_hf_config = AutoConfig.from_pretrained(runtime_conf.actor_conf_path)

        actor_num_layers, actor_hidden_size, actor_vocab_size = get_hf_configs(actor_hf_config)

        gpus_per_model = torch.distributed.get_world_size()

        seq_length = model_conf.max_answer_seq_len + model_conf.max_prompt_seq_len
        # batch_size = args.per_device_generation_batch_size * args.generation_batches * args.ppo_epochs * gpus_per_model * 1 if args.unsupervised_dataset_name is None else 2
        # total batch_size
        batch_size = train_conf.data_conf.batch_size * train_conf.rollout_size * train_conf.rl_train_epochs * gpus_per_model * 1 if train_conf.data_conf.unsupervised_dataset_name is None else 2
        samples_per_second = batch_size / e2e_time

        # 1+ 2(bwd)
        actor_checkpoint_activations_factor = 4 if runtime_conf.actor_conf.gradient_checkpointing else 3

        if getattr(model_conf, 'lora_conf', None) is not None and model_conf.lora_conf.actor_lora_dim > 0:
            # flops 来说，-一次大的bwd，+2次 小kernel
            k = model_conf.lora_conf.actor_lora_dim * 2 / actor_hidden_size
            actor_checkpoint_activations_factor -= (1 - k)

        actor_num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel() for p in actor_model.parameters()
        ]) if actor_model is not None else (12 * actor_num_layers * actor_hidden_size * actor_hidden_size +
                                            actor_vocab_size * actor_hidden_size)
        actor_params_in_billions = actor_num_params / (1e9)

        # Megatron paper's formula to calculate training flops

        actor_train_flops_per_iteration = calculate_flops(actor_checkpoint_activations_factor, batch_size, seq_length,
                                                          actor_hf_config)

        total_train_flops = actor_train_flops_per_iteration
        if not is_ac_share:

            # Critic model passed here is  a DeepSpeed Engine. The module inside is the Reward model (that wraps a HF model).

            critic_hf_config = critic_model.module.config if critic_model is not None else AutoConfig.from_pretrained(
                runtime_conf.critic_conf_path)

            critic_num_layers, critic_hidden_size, critic_vocab_size = get_hf_configs(critic_hf_config)

            critic_num_params = sum([
                p.ds_numel if hasattr(p, "ds_tensor") else p.numel() for p in critic_model.parameters()
            ]) if critic_model is not None else (12 * critic_num_layers * critic_hidden_size * critic_hidden_size +
                                                 critic_vocab_size * critic_hidden_size)
            critic_params_in_billions = critic_num_params / (1e9)

            critic_checkpoint_activations_factor = 4 if runtime_conf.critic_conf.gradient_checkpointing else 3
            if getattr(model_conf, 'lora_conf', None) is not None and model_conf.lora_conf.critic_lora_dim > 0:
                k = model_conf.lora_conf.critic_lora_dim * 2 / critic_hidden_size
                critic_checkpoint_activations_factor -= (1 - k)

            critic_train_flops_per_iteration = calculate_flops(critic_checkpoint_activations_factor, batch_size,
                                                               seq_length, critic_hf_config)
            total_train_flops += critic_train_flops_per_iteration

        train_tflops = total_train_flops / (train_time * gpus_per_model * (10**12))

        gen_bs = train_conf.data_conf.batch_size * gpus_per_model

        # Modified formula for calculating flops in the forward pass only
        gen_flops_per_iteration = (24 * gen_bs * seq_length * actor_num_layers *
                                   (actor_hidden_size**2)) * (1.0 + (seq_length / (6.0 * actor_hidden_size)) +
                                                              (actor_vocab_size /
                                                               (16.0 * actor_num_layers * actor_hidden_size)))

        # gen_exp_time传入的是gen一个batch_size的时间
        gen_tflops = gen_flops_per_iteration / (gen_exp_time * gpus_per_model * (10**12))

        total_gen_flops = gen_flops_per_iteration * train_conf.rollout_size

        if actor_hf_config.torch_dtype == torch.float16:
            num_bytes = 2
        elif actor_hf_config.torch_dtype == torch.float32:
            num_bytes = 4
        else:
            num_bytes = -1

        pertok_lat = gen_exp_time / model_conf.max_answer_seq_len
        gen_bw = 1 / pertok_lat * actor_num_params * num_bytes / 1e9

        total_flops_per_iteration = total_train_flops + total_gen_flops
        # 单卡概念
        total_tflops = total_flops_per_iteration / (e2e_time * gpus_per_model * (10**12))

        print(
            f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {total_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
        )
        print(
            f"Per Step FLOPS => TOTAL_TFLOPS: {total_flops_per_iteration / 10**12:.2f}, TRAIN_TFLOPS: {total_train_flops / 10**12:.2f}, "
            f" GEN_FLOPS: {total_gen_flops / 10**12:.2f} train_total_bs: {batch_size}, gen_total_bs: {gen_bs * train_conf.rollout_size}"
        )
        print(
            f"Generation => Latency: {gen_exp_time:.2f}s, Per-token Latency {pertok_lat*1000:.2f} ms, TFLOPs: {gen_tflops:.2f}, BW: {gen_bw if num_bytes > 0 else num_bytes:.2f} GB/sec, Answer Seq. Length: {model_conf.max_answer_seq_len}"
        )
        print(f"Training   => Latency: {train_time:.2f}s, TFLOPs: {train_tflops:.2f}")
        actor_param_string = f"{actor_params_in_billions:.3f} B" if actor_params_in_billions != 0 else "NA"

        print(f"Actor Model Parameters => {actor_param_string}")
        if not is_ac_share:
            critic_param_string = f"{critic_params_in_billions:.3f} B" if critic_params_in_billions != 0 else "NA"
            print(f"Critic Model Parameters => {critic_param_string}")

        print(
            "MemAllocated={}GB, MaxMemAllocated={}GB".format(
                round(torch.cuda.memory_allocated() / 1024**3, 2),
                round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            ))        


# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_flops(checkpoint_activations_factor, batch_size, seq_length, hf_config):
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size * seq_length * num_layers *
                           (hidden_size**2)) * (1.0 + (seq_length / (6.0 * hidden_size)) +
                                                (vocab_size / (16.0 * num_layers * hidden_size)))
    return flops_per_iteration


def get_hf_configs(hf_config):
    num_layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", None))
    hidden_size = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)

    assert all((num_layers, hidden_size,
                vocab_size)), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size


TRAIN_TFLOPS = None
GEN_TFLOPS = None
E2E_TFLOPS = None


# Enhanced version of the function above that provides calculations and printing for Step 3
def print_throughput_step3_sep(e2e_time=None,
                               gen_exp_time=None,
                               actor_train_time=None,
                               critic_train_time=None):
    # model_conf = global_context().model_conf
    # train_conf = global_context().train_conf
    runtime_conf = global_context().runtime_conf
    placement = global_context().placement

    actor_sep_config = runtime_conf.rlhf_sep_config.actor_sep_config.megatron_args
    critic_sep_config = runtime_conf.rlhf_sep_config.critic_sep_config.megatron_args
    tokenizer = AutoConfig.from_pretrained(actor_sep_config['vocab_file'])

    actor_num_layers, actor_hidden_size, vocab_size = actor_sep_config['num_layers'], actor_sep_config[
        'hidden_size'], tokenizer.vocab_size
    
    actor_checkpointing = runtime_conf.rlhf_sep_config.actor_sep_config.checkpointing
    critic_checkpointing = runtime_conf.rlhf_sep_config.critic_sep_config.checkpointing

    gpus_per_model = torch.distributed.get_world_size()

    seq_length = runtime_conf.rlhf_sep_config.seq_length
    # batch_size = args.per_device_generation_batch_size * args.generation_batches * args.ppo_epochs * gpus_per_model * 1 if args.unsupervised_dataset_name is None else 2
    # total batch_size
    batch_size = runtime_conf.rlhf_sep_config.train_global_batch_size
    

    # 1+ 2(bwd)
    actor_checkpoint_activations_factor = 4 if actor_checkpointing else 3

    lora_dim = os.environ.get('LORA_DIM', None)

    if lora_dim:
        lora_dim = int(lora_dim)
        # flops 来说，-一次大的bwd，+2次 小kernel
        k = lora_dim * 2 / actor_hidden_size
        actor_checkpoint_activations_factor -= (1 - k)

    actor_num_params = (12 * actor_num_layers * actor_hidden_size * actor_hidden_size +
                                        vocab_size * actor_hidden_size)
    actor_params_in_billions = actor_num_params / (1e9)

    # Megatron paper's formula to calculate training flops

    actor_train_flops_per_iteration = (24 * actor_checkpoint_activations_factor * batch_size * seq_length * actor_num_layers *
                        (actor_hidden_size**2)) * (1.0 + (seq_length / (6.0 * actor_hidden_size)) +
                                            (vocab_size / (16.0 * actor_num_layers * actor_hidden_size)))        

    total_train_flops = actor_train_flops_per_iteration


    # Critic model passed here is  a DeepSpeed Engine. The module inside is the Reward model (that wraps a HF model).

    critic_num_layers, critic_hidden_size, vocab_size = critic_sep_config['num_layers'], critic_sep_config[
        'hidden_size'], tokenizer.vocab_size

    critic_num_params = (12 * critic_num_layers * critic_hidden_size * critic_hidden_size +
                                            vocab_size * critic_hidden_size)
    critic_params_in_billions = critic_num_params / (1e9)

    critic_checkpoint_activations_factor = 4 if critic_checkpointing else 3
    if lora_dim:
        k = lora_dim * 2 / critic_hidden_size
        critic_checkpoint_activations_factor -= (1 - k)

    critic_train_flops_per_iteration = (24 * critic_checkpoint_activations_factor * batch_size * seq_length * critic_num_layers *
                        (critic_hidden_size**2)) * (1.0 + (seq_length / (6.0 * critic_hidden_size)) +
                                            (vocab_size / (16.0 * critic_num_layers * critic_hidden_size)))      
    total_train_flops += critic_train_flops_per_iteration


    model_ranks = placement.model_ranks
    actor_gpu_nums, critic_gpu_nums = len(model_ranks.actor_ranks), len(model_ranks.critic_ranks)
    pred_actor_gpu_nums = len(model_ranks.pred_actor_ranks)
    if actor_train_time is not None:
        actor_train_tflops = actor_train_flops_per_iteration / (actor_train_time * actor_gpu_nums * (10**12))
        print(f"Training   => Actor Latency: {actor_train_time:.2f}s, TFLOPs: {actor_train_tflops:.2f}")            
        
    if critic_train_time is not None:
        critic_train_tflops = actor_train_flops_per_iteration / (critic_train_time * critic_gpu_nums * (10**12))
        print(f"Training   => Critic Latency: {critic_train_time:.2f}s, TFLOPs: {critic_train_tflops:.2f}")            

    gen_bs = runtime_conf.rlhf_sep_config.rollout_batch_size * pred_actor_gpu_nums

    # Modified formula for calculating flops in the forward pass only
    gen_flops_per_iteration = (24 * gen_bs * seq_length * actor_num_layers *
                                (actor_hidden_size**2)) * (1.0 + (seq_length / (6.0 * actor_hidden_size)) +
                                                            (vocab_size /
                                                            (16.0 * actor_num_layers * actor_hidden_size)))

    actor_param_string = f"{actor_params_in_billions:.3f} B" if actor_params_in_billions != 0 else "NA"
    print(f"Actor Model Parameters => {actor_param_string}")

    critic_param_string = f"{critic_params_in_billions:.3f} B" if critic_params_in_billions != 0 else "NA"
    print(f"Critic Model Parameters => {critic_param_string}")        
    total_gen_flops = gen_flops_per_iteration * runtime_conf.rlhf_sep_config.batches_per_rollout

    total_flops_per_iteration = total_train_flops + total_gen_flops

    print(
        f"Per Step FLOPS => TOTAL_TFLOPS: {total_flops_per_iteration / 10**12:.2f}, TRAIN_TFLOPS: {total_train_flops / 10**12:.2f}, "
        f" GEN_FLOPS: {total_gen_flops / 10**12:.2f} train_total_bs: {batch_size}, gen_total_bs: {gen_bs * runtime_conf.rlhf_sep_config.batches_per_rollout}"
    )        
    # if actor_hf_config.torch_dtype == torch.float16:
    #     num_bytes = 2
    # elif actor_hf_config.torch_dtype == torch.float32:
    #     num_bytes = 4
    # else:
    #     num_bytes = -1
    num_bytes = 2    

    if e2e_time is not None:
        """这里暂时不考虑pipeline，先把时间打出来了。。
        考虑pipeline的话，时间肯定非严格对齐。generate_batch_size 之类的也不一致
        """
        


        samples_per_second = batch_size / e2e_time
        
        


        # 单卡概念
        total_tflops = total_flops_per_iteration / (e2e_time * gpus_per_model * (10**12))
        print(
            f"End-to-End => Latency: {e2e_time:.2f}s, TFLOPs: {total_tflops:.2f}, Samples/sec: {samples_per_second:.2f}"
            f", Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Total Seq. Length: {seq_length}"
        )


    if gen_exp_time is not None:
        # gen_exp_time传入的是gen一个batch_size的时间
        gen_tflops = gen_flops_per_iteration / (gen_exp_time * gpus_per_model * (10**12))
        pertok_lat = gen_exp_time / runtime_conf.rlhf_sep_config.max_new_tokens
        gen_bw = 1 / pertok_lat * actor_num_params * num_bytes / 1e9
        print(
            f"Generation => Latency: {gen_exp_time:.2f}s, Per-token Latency {pertok_lat*1000:.2f} ms, TFLOPs: {gen_tflops:.2f}, BW: "
            f"{gen_bw if num_bytes > 0 else num_bytes:.2f} GB/sec, Answer Seq. Length: {runtime_conf.rlhf_sep_config.max_new_tokens}"
        )            

    # self.logging(
    #     "epoch={}/micro_step={}/global_step={}, RunningAvgSamplesPerSec={}, CurrSamplesPerSec={}, "
    #     "MemAllocated={}GB, MaxMemAllocated={}GB".format(
    #         self.epoch_count,
    #         self.micro_step_count,
    #         self.global_step_count,
    #         self.avg_samples_per_sec(),
    #         self.batch_size / self.step_elapsed_time,
    #         round(get_accelerator().memory_allocated() / 1024**3, 2),
    #         round(get_accelerator().max_memory_allocated() / 1024**3, 2),
    #     ))
    print(
        "MemAllocated={}GB, MaxMemAllocated={}GB".format(
            round(torch.cuda.memory_allocated() / 1024**3, 2),
            round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        ))        