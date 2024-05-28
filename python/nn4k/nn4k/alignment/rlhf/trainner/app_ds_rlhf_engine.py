# coding: utf-8

import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from transformers import get_scheduler
from alignment.rlhf.utils.ds_utils import get_train_ds_config, get_eval_ds_config
from alignment.rlhf.utils.utils import get_optimizer_grouped_parameters, log_init
from alignment.util.global_vars import global_context

class SetModelHooks(object):
    def __init__(self, current_mdoel):
        self._current_model = current_mdoel


    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            from alignment.rlhf.distributed.distributed_rlhf_engine import CallModelHooks
            with CallModelHooks('on_model_init', self._current_model) as model_hook:
                module = func(*args, **kwargs)
                model_hook.set_module(module)
            return module

        return wrapped_func


class DeepSpeedSFTEngine:

    def __init__(self, model_provider, num_total_iters, context=None, init_models=True):
        """ Deep speed SFT Engine.

        Args:
            model_provider: The model provider.
            num_total_iters: The total number of iterations.
        """
        self.num_total_iters = num_total_iters
        self.model_provider = model_provider
        self.context = context or global_context()

        if init_models:
            self.model = self.init_model()
        else:
            self.model = None

    def init_model(self):
        stime = log_init("SFT")

        train_conf = self.context.train_conf
        ds_conf = self.context.runtime_conf.dist_conf
        hybrid_engine_conf = ds_conf.hybrid_engine_conf
        mini_train_batch_size = train_conf.data_conf.batch_size
        gradient_accumulation_steps = train_conf.actor_optimizer.gradient_accumulation_steps
        ds_config = get_train_ds_config(
            offload=ds_conf.offload,
            stage=ds_conf.zero_stage,
            enable_hybrid_engine=hybrid_engine_conf.enable_hybrid_engine,
            inference_tp_size=hybrid_engine_conf.inference_tp_size,
            release_inference_cache=hybrid_engine_conf.release_inference_cache,
            pin_parameters=(not hybrid_engine_conf.unpin_actor_parameters),
            tp_gather_partition_size=hybrid_engine_conf.tp_gather_partition_size
        )
        ds_config['train_micro_batch_size_per_gpu'] = mini_train_batch_size
        # TODO(jeff): we should probably set grad accumulation steps here as well for clarity
        ds_config['train_batch_size'] = mini_train_batch_size * \
                                        torch.distributed.get_world_size() * \
                                        gradient_accumulation_steps

        model = self.model_provider.get_model(self.context.model_conf, ds_config=ds_config)

        # Optimizer
        if train_conf.optimizer and train_conf.optimizer.instance:
            optim = train_conf.optimizer.instance
        else:
            AdamOptimizer = DeepSpeedCPUAdam if ds_conf.offload else FusedAdam
            optim_params = get_optimizer_grouped_parameters(
                model, train_conf.optimizer.weight_decay)
            assert not all([len(optim_param["params"]) == 0 for optim_param in optim_params]), "No parameter to opt!"
            optim = AdamOptimizer(optim_params,
                                  lr=train_conf.optimizer.learning_rate,
                                  betas=(0.9, 0.95))

        # LR Scheduler
        scheduler_conf = train_conf.optimizer.scheduler_conf
        lr_scheduler = get_scheduler(
            name=scheduler_conf.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=scheduler_conf.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        engine, *_ = deepspeed.initialize(model=model,
                                          optimizer=optim,
                                          lr_scheduler=lr_scheduler,
                                          config=ds_config)

        log_init("SFT", stime=stime)

        return engine


class DeepSpeedACEngine:

    def __init__(self, model_provider, num_total_iters, context=None, init_models=True):
        """ Deep speed AC Engine.

        Args:
            model_provider: The model provider.
            num_total_iters: The total number of iterations.
        """
        self.num_total_iters = num_total_iters
        self.model_provider = model_provider
        self.context = context or global_context()

        if init_models:
            self.init_all_models()
        else:
            self.actor = None
            self.ref = None
            self.reward = None
            self.critic = None

    def init_all_models(self):
        self.actor = self.init_actor()
        if self.context.model_conf.enable_ref:
            self.ref = self.init_ref(self.actor)
        if self.context.model_conf.enable_ema:
            self.actor_ema = self.init_ema()
        self.reward = self.init_reward()

    @SetModelHooks("actor")
    def init_actor(self):
        stime = log_init("Actor")

        train_conf = self.context.train_conf
        actor_ds_conf = self.context.runtime_conf.actor_conf
        hybrid_engine_conf = actor_ds_conf.hybrid_engine_conf
        mini_train_batch_size = train_conf.rl_train_batch_size
        gradient_accumulation_steps = train_conf.actor_optimizer.gradient_accumulation_steps
        ds_config = get_train_ds_config(
            offload=actor_ds_conf.offload,
            stage=actor_ds_conf.zero_stage,
            enable_hybrid_engine=hybrid_engine_conf.enable_hybrid_engine,
            inference_tp_size=hybrid_engine_conf.inference_tp_size,
            release_inference_cache=hybrid_engine_conf.release_inference_cache,
            pin_parameters=(not hybrid_engine_conf.unpin_actor_parameters),
            tp_gather_partition_size=hybrid_engine_conf.tp_gather_partition_size
        )
        ds_config['train_micro_batch_size_per_gpu'] = mini_train_batch_size
        # TODO(jeff): we should probably set grad accumulation steps here as well for clarity
        ds_config['train_batch_size'] = mini_train_batch_size * \
                                        torch.distributed.get_world_size() * \
                                        gradient_accumulation_steps

        actor_model = self.model_provider.get_actor_model(self.context.model_conf, ds_config=ds_config)

        # print("actor_model is : {}".format(actor_model))

        # Optimizer
        if train_conf.actor_optimizer and train_conf.actor_optimizer.instance:
            optim = train_conf.actor_optimizer.instance
        else:
            AdamOptimizer = DeepSpeedCPUAdam if actor_ds_conf.offload else FusedAdam
            optim_params = get_optimizer_grouped_parameters(
                actor_model, train_conf.actor_optimizer.weight_decay)
            assert not all([len(optim_param["params"]) == 0 for optim_param in optim_params]), "No parameter to opt!"
            optim = AdamOptimizer(optim_params,
                                  lr=train_conf.actor_optimizer.learning_rate,
                                  betas=(0.9, 0.95))

        # LR Scheduler
        scheduler_conf = train_conf.actor_optimizer.scheduler_conf
        lr_scheduler = get_scheduler(
            name=scheduler_conf.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=scheduler_conf.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim,
                                                lr_scheduler=lr_scheduler,
                                                config=ds_config)

        log_init("Actor", stime=stime)

        return actor_engine

    @SetModelHooks("reward")
    def init_reward(self):
        stime = log_init("Reward")
        # DS Config
        train_conf = self.context.train_conf
        critic_ds_conf = self.context.runtime_conf.critic_conf or self.context.runtime_conf.actor_conf
        zero_stage = critic_ds_conf.zero_stage
        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0

        mini_train_batch_size = train_conf.rl_train_batch_size
        gradient_accumulation_steps = train_conf.actor_optimizer.gradient_accumulation_steps
        ds_config = get_eval_ds_config(offload=critic_ds_conf.offload,
                                       stage=zero_stage)
        ds_config['train_micro_batch_size_per_gpu'] = mini_train_batch_size
        ds_config['train_batch_size'] = mini_train_batch_size * \
                                        torch.distributed.get_world_size() * \
                                        gradient_accumulation_steps

        # ds_eval_config = get_eval_ds_config(offload=False, stage=0)

        def _create_reward_model_fn():
            return self.model_provider.get_reward_model(ds_config=ds_config)

        reward_policy = self.context.runtime_conf.reward_policy
        if reward_policy is None:
            reward_model = _create_reward_model_fn()
            reward_engine, *_ = deepspeed.initialize(model=reward_model, config=ds_config)
        else:
            from alignment.rlhf.trainner.mixed_model import RewardModel
            from transformers.models.auto import AutoConfig
            config = AutoConfig.from_pretrained(self.context.model_conf.model_provider.reward_model_path)
            reward_engine = RewardModel(reward_policy, _create_reward_model_fn, config)

        log_init("Reward", stime=stime)
        return reward_engine

    @SetModelHooks("ref")
    def init_ref(self, actor):
        stime = log_init("Ref")

        frozen_layer_conf = self.context.model_conf.frozen_layer_conf
        if actor and frozen_layer_conf:
            # todo ===== bug...
            return actor

        train_conf = self.context.train_conf
        actor_ds_conf = self.context.runtime_conf.actor_conf
        mini_train_batch_size = train_conf.rl_train_batch_size
        gradient_accumulation_steps = train_conf.actor_optimizer.gradient_accumulation_steps

        zero_stage = actor_ds_conf.zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
            zero_stage = 0
        ds_config = get_eval_ds_config(self.context.runtime_conf.offload_reference_model,
                                       zero_stage)
        ds_config['train_micro_batch_size_per_gpu'] = mini_train_batch_size
        # TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config['train_batch_size'] = mini_train_batch_size * \
                                        torch.distributed.get_world_size() * \
                                        gradient_accumulation_steps

        def _create_ref_model_fn():
            return self.model_provider.get_actor_model(self.context.model_conf, ds_config=ds_config)

        ref_policy = self.context.runtime_conf.ref_policy
        if ref_policy is None:
            ref_model = _create_ref_model_fn()
            ref_engine, *_ = deepspeed.initialize(model=ref_model, config=ds_config)
        else:
            from alignment.rlhf.trainner.mixed_model import RefModel
            from transformers.models.auto import AutoConfig
            config = AutoConfig.from_pretrained(self.context.model_conf.model_provider.initial_model_path)
            ref_engine = RefModel(ref_policy, _create_ref_model_fn, config)

        log_init("Ref", stime=stime)
        return ref_engine

    def init_ema(self):
        stime = log_init("EMA")
        # DS Config
        train_conf = self.context.train_conf
        runtime_conf = self.context.runtime_conf
        zero_stage = runtime_conf.actor_conf.zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
        ds_config = get_eval_ds_config(runtime_conf.offload_reference_model,
                                       zero_stage)

        actor_model_ema = self.model_provider.get_ema_model(model_conf=self.context.model_conf, ds_config=ds_config)

        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config)

        log_init("EMA", stime=stime)
        return ema_engine


class DeepSpeedACShareEngine(DeepSpeedACEngine):
    """Share actor-critic engine for RLHF"""

    def __init__(self, model_provider, num_total_iters, context=None, init_models=True):
        super().__init__(model_provider, num_total_iters, context, init_models)


class DeepSpeedACNoneShareEngine(DeepSpeedACEngine):

    def __init__(self, model_provider, num_total_iters, context=None, init_models=True):
        super().__init__(model_provider, num_total_iters, context, init_models=init_models)

    def init_all_models(self):
        super(DeepSpeedACNoneShareEngine, self).init_all_models()
        self.critic = self.init_critic()
    
    @SetModelHooks('critic')
    def init_critic(self):
        stime = log_init("Critic")

        train_conf = self.context.train_conf
        critic_ds_conf = self.context.runtime_conf.critic_conf
        mini_train_batch_size = train_conf.rl_train_batch_size
        gradient_accumulation_steps = train_conf.critic_optimizer.gradient_accumulation_steps

        ds_config = get_train_ds_config(offload=critic_ds_conf.offload,
                                        stage=critic_ds_conf.zero_stage)
        ds_config['train_micro_batch_size_per_gpu'] = mini_train_batch_size
        # TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config['train_batch_size'] = mini_train_batch_size * torch.distributed.get_world_size() * \
                                        gradient_accumulation_steps

        # TODO(jeff): should not be needed, we should be able to use ds_config above
        # TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
        # ds_eval_config = get_eval_ds_config(offload=False, stage=0)

        critic_model = self.model_provider.get_critic_model(model_conf=self.context.model_conf,
                                                            ds_config=ds_config)

        # Optimizer
        critic_optimizer = train_conf.critic_optimizer
        if critic_optimizer and critic_optimizer.instance:
            optim = critic_optimizer.instance
        else:
            AdamOptimizer = DeepSpeedCPUAdam if critic_ds_conf.offload else FusedAdam
            optim_params = get_optimizer_grouped_parameters(
                critic_model, critic_optimizer.weight_decay)
            optim = AdamOptimizer(optim_params,
                                  lr=critic_optimizer.learning_rate,
                                  betas=(0.9, 0.95))
        # LR Scheduler
        scheduler_conf = critic_optimizer.scheduler_conf
        lr_scheduler = get_scheduler(
            name=scheduler_conf.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=scheduler_conf.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)

        if critic_ds_conf.gradient_checkpointing:
            critic_engine.gradient_checkpointing_enable()

        log_init("Critic", stime=stime)
        return critic_engine

class DeepSpeedACNoneShareSEPEngine(DeepSpeedACNoneShareEngine):
    """Share actor-critic engine for RLHF"""
    def __init__(self, model_provider, num_total_iters, context=None, init_models=True):
        super().__init__(model_provider, num_total_iters, context, init_models)

        self.pred_actor = None
        self.pred_critic = None

    def init_all_models(self):
        super().init_all_models()
        self.pred_actor = self.init_pred_actor()
        self.pred_critic = self.init_pred_critic()

    @SetModelHooks('pred_actor')
    def init_pred_actor(self):
        actor_model = self.model_provider.get_pred_actor_model(self.context.model_conf, ds_config={})
        return actor_model
            
    
    @SetModelHooks('pred_critic')
    def init_pred_critic(self):
        pred_critic_model = self.model_provider.get_pred_critic_model(model_conf=self.context.model_conf, ds_config={})
        return pred_critic_model    


    @SetModelHooks('critic')
    def init_critic(self):
        critic_model = self.model_provider.get_critic_model(model_conf=self.context.model_conf, ds_config={})
        return critic_model

    @SetModelHooks("ref")
    def init_ref(self, actor):
        return self.model_provider.get_initial_model(self.context.model_conf, ds_config={})


    @SetModelHooks("reward")
    def init_reward(self):
        return self.model_provider.get_reward_model(ds_config={})

    @SetModelHooks("actor")
    def init_actor(self):
        actor_model = self.model_provider.get_actor_model(self.context.model_conf, ds_config={})
        return actor_model