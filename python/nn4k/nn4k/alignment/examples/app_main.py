from alignment.api.rlhf.config import ModelConfig, TrainConfig, DataConfig, LoraConfig, \
    EngineDistributedConfig, SchedulerConfig, OptimizerConfig, RuntimeConfig, \
    DistributedConfig, Placement, EvalConfig, AllCollocate, ModelRanks, InitialRewardSeparate, AllSeparate
from alignment.model.default_model_impl import DefaultACNoneShareModelProvider


import os


def get_from_env(param_name, default_val):
    return_val = default_val
    try:
        return_val = eval(os.environ.get(param_name))
    except:
        pass
    return return_val


print(f'cur env: {os.environ}')

BUCKET_SIZE = int(5e8 / 10)    # for reduce

llama_conf = dict(
    llama_7b=dict(sft_model_path='huggyllama.llama-7b/main/20230612181113/hf_model',
                  reward_model_path='paper/models/output_step2_llama-7b_epoch1_lr9.65e-6'),
    llama_13b=dict(
        sft_model_path='huggyllama.llama-13b/main/20230615140156/hf_model',
        reward_model_path='paper/models/output_step2_llama-13b_epoch1_lr9.65e-6'),
    llama_33b=dict(
        sft_model_path='huggyllama.llama-30b/main/20230831180440/hf_model',
        reward_model_path='paper/models/output_step2_llama-30b_epoch1_lr9.65e-6'),
    llama_65b=dict(
        sft_model_path='huggyllama.llama-65b/main/20230831152814/hf_model',
        reward_model_path='paper/models/output_step2_llama-65b_epoch1_lr9.65e-6'))


def train_ac_share(need_value_head=True):
    from model import CustomerChatModelProvider
    train(CustomerChatModelProvider, need_value_head=need_value_head)


def train_ac_no_share(need_value_head=False):
    train(DefaultACNoneShareModelProvider, need_value_head=need_value_head)


def train(model_provider_cls, need_value_head):
    # ac non share时，不需要value_head
    enable_hybrid_engine = os.environ.get('ENABLE_HYBRID_ENGINE', 'False') == 'TRUE'
    act_zero_stage = get_from_env('ACTOR_ZERO_STAGE', 3)
    cri_stage = get_from_env('CRITIC_ZERO_STAGE', 3)

    batch_size = get_from_env('ROLLOUT_BATCH_SIZE', 1)
    rollout_size = get_from_env('BATCHES_PER_ROLLOUT', 1)
    act_checkpointing = os.environ.get('CHECKPOINTING', 'False') == 'TRUE'

    rl_train_batch_size = get_from_env('PER_DEVICE_TRAIN_BATCH_SIZE', 1)
    # for hybrid engine
    release_inference_cache = os.environ.get('RELEASE_INFER_CACHE', 'False') == 'TRUE'

    offload_actor_model = os.environ.get('OFFLOAD_ACTOR_MODEL', 'False') == 'TRUE'
    offload_critic_model = os.environ.get('OFFLOAD_CRITIC_MODEL', 'False') == 'TRUE'

    run_ref_reward_async = os.environ.get('RUN_REF_REWARD_ASYNC', 'False') == 'TRUE'

    deploy_path = f"llama_dist_output_new2/cri_{cri_stage}_pred_bs{batch_size}_roll_{rollout_size}_trainbs_{rl_train_batch_size}_hybrid_{enable_hybrid_engine}_release_inference_cache{release_inference_cache}_BUCKET_SIZE{BUCKET_SIZE}_act_checkpointing{act_checkpointing}"
    print(f'deploy_path: {deploy_path}')

    sft_model_path = llama_conf['sft_model_path']
    reward_model_path = llama_conf['reward_model_path']

    llama_paths = llama_conf[os.environ.get('LLAMA_CONF', 'llama_7b')]

    model_provider = model_provider_cls(sft_model_path, reward_model_path)
    # frozen here
    actor_gen_kwargs = dict(
        min_length=480,
        max_length=480,
    )
    model_conf = ModelConfig(
        model_provider=model_provider,
        lora_conf=LoraConfig(actor_lora_dim=128, critic_lora_dim=128),
        enable_ema=False,
        need_value_head=need_value_head,
        run_ref_reward_async=run_ref_reward_async,
        actor_gen_kwargs=actor_gen_kwargs,
    )
    data_conf = DataConfig(batch_size=batch_size,
                           dataset=['Dahoas/rm-static'],
                           workdir='datasets'
    # data_range=(0, 0.001)
                           )

    train_conf = TrainConfig(
        data_conf=data_conf,
        num_train_epochs=1000000,
        rollout_size=rollout_size,
        save_checkpoints_steps=500000,
        rl_train_batch_size=rl_train_batch_size,
        actor_optimizer=OptimizerConfig(
            learning_rate=9.65e-12,
            weight_decay=0.1,
            gradient_accumulation_steps=1,
            scheduler_conf=SchedulerConfig(lr_scheduler_type='cosine', num_warmup_steps=100)),
        critic_optimizer=OptimizerConfig(
            learning_rate=5e-12,
            weight_decay=0,
            gradient_accumulation_steps=1,
            scheduler_conf=SchedulerConfig(lr_scheduler_type='cosine', num_warmup_steps=100)),
    )
    runtime_conf = RuntimeConfig(
        seed=10,
        offload_reference_model=False,
        dtype="fp16",
        zero_opt_param={
            "allgather_bucket_size": BUCKET_SIZE,
            "reduce_bucket_size": BUCKET_SIZE
        },
        actor_conf_path=sft_model_path,
        critic_conf_path=reward_model_path,
        skip_load=True,
        actor_conf=DistributedConfig(
            zero_stage=act_zero_stage,
            gradient_checkpointing=act_checkpointing,
            offload=offload_actor_model,
            hybrid_engine_conf=EngineDistributedConfig(enable_hybrid_engine=enable_hybrid_engine,
                                                       unpin_actor_parameters=False,
                                                       release_inference_cache=release_inference_cache,
                                                       inference_tp_size=1,
                                                       tp_gather_partition_size=8),
            enable_init_partition=True,
        ),
        critic_conf=DistributedConfig(
            zero_stage=cri_stage,
            gradient_checkpointing=act_checkpointing,
            offload=offload_critic_model,
            hybrid_engine_conf=EngineDistributedConfig(enable_hybrid_engine=False,
                                                       unpin_actor_parameters=False,
                                                       release_inference_cache=False,
                                                       inference_tp_size=1,
                                                       tp_gather_partition_size=8),
            enable_init_partition=True,
        ))
    eval_conf = EvalConfig(data_conf=data_conf, eval_at_first=False, eval_every_steps=2000, steps=10)


    placement_strategy = os.environ.get('PLACEMENT_STRATEGY', "flattening").lower()
    placement = {
        'flattening': AllCollocate(),
        'interleaving':InitialRewardSeparate(),
        'separation': AllSeparate()
    }[placement_strategy]

    from alignment.rlhf.trainner.exec_engine import APPExecutionEngine
    engine = APPExecutionEngine(
        model_conf=model_conf,
        train_conf=train_conf,
        runtime_conf=runtime_conf,
        placement= placement
    )
    engine.run()


if __name__ == "__main__":

    if get_from_env('AC_MODE', 'SHARE') == 'SHARE':
        train_ac_share()    # with value_head
    else:
        train_ac_no_share()
