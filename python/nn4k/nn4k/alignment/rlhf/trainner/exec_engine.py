import random
import deepspeed
import torch
import inspect
import gc
import time

from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer

from alignment.util.global_vars import global_context
from alignment.rlhf.config import ModelConfig, TrainConfig, EvalConfig, RuntimeConfig, \
    Placement, AllCollocate, ProfilerConfig
from alignment.api.rlhf.model_provider import ACNoneShareModelProvider, ACShareModelProvider, ACNoneShareSEPModelProvider
from alignment.api.utils import dist_util
from alignment.rlhf.data.default_data_impl import create_datasets
from alignment.rlhf.data.data_utils import MiniDataset, RolloutStorage
from alignment.rlhf.distributed.distributed_rlhf_engine import DistributedDeepSpeedACEngine
from alignment.rlhf.distributed.distributed_rlhf_sep_engine import DistributedDeepSpeedSEPACEngine
from alignment.rlhf.distributed.model_placement import ModelPlacement
from alignment.rlhf.hooks import LoggingHook
from alignment.rlhf.module.util import assign_module
from alignment.rlhf.trainner.app_ds_rlhf_engine import DeepSpeedACShareEngine, DeepSpeedACNoneShareEngine, DeepSpeedACNoneShareSEPEngine
from alignment.rlhf.module.ac_none_share_module import ACNoneShareDeepSpeedSEPModule
from alignment.rlhf.utils.save_utils import load_rlhf_checkpoint
from alignment.rlhf.utils.utils import print_rank_0, to_device, set_random_seed, get_all_reduce_mean, \
    moving_average, get_dynamic_port
from alignment.app.util import logger
from alignment.rlhf.utils.perf import print_throughput_step3
from alignment.rlhf.module.ac_share_module import ACShareDeepSpeedModule

hooks = []


def _call_hooks(func, **kwargs):
    if isinstance(func, str):
        func = [func]
    for hook in hooks:
        try:
            for f in func:
                getattr(hook, f)(**kwargs)
        except:
            logger.error(f"Failed to call {hook}.{f} with {kwargs}")
            raise


def _is_on_policy(ctx):
    return ctx.self.train_conf.rl_policy.lower() == "on"


def next_data(loader):
    for dat in loader:
        yield dat


class APPExecutionEngine:
    def __init__(self, 
                 model_conf: ModelConfig,
                 train_conf: TrainConfig,
                 runtime_conf: RuntimeConfig,
                 eval_conf: Optional[EvalConfig] = None,
                 placement: Placement = AllCollocate(), #
                 profile_config: ProfilerConfig = None,
                 ):
        """
        :param model_conf:
        :param train_conf:
        :param runtime_conf:
        :param eval_conf:
        :param placement: placement ratio. support three distributed strategies:
        1. Flattening: AllCollocate()
        2. Interleaving: InitialRewardSeparate()
        3. Separation: AllSeparate()
        :param profile_config:
        """

        self.model_conf = model_conf
        self.train_conf = train_conf
        self.runtime_conf = runtime_conf
        self.eval_conf = eval_conf
        self.placement = placement
        self.profile_config = profile_config
        self.context = global_context()

        if self.train_conf.rl_train_batch_size is None:
            self.train_conf.rl_train_batch_size = self.train_conf.data_conf.batch_size
        setattr(self.context, "model_conf", model_conf)
        setattr(self.context, "self.train_conf", self.train_conf)
        setattr(self.context, "eval_conf", eval_conf)
        setattr(self.context, "runtime_conf", runtime_conf)
        setattr(self.context, "profile_conf", profile_config)

        local_rank = dist_util.get_local_rank()
        if local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            deepspeed.init_distributed(distributed_port=get_dynamic_port(runtime_conf.custom_port))

        setattr(self.context, "local_rank", local_rank)
        self.global_rank = torch.distributed.get_rank()

        # assert not runtime_conf.actor_conf.offload, "zero-offload is not currently supported but coming soon!"
        assert model_conf.model_provider, "Missing model_provider in model config."
        model_provider = model_conf.model_provider
        print("use model_provider of : {}".format(model_provider))

        self.unsupervised_training_enabled = self.train_conf.data_conf.unsupervised_dataset_name and \
                                        self.train_conf.data_conf.unsupervised_dataset_config_name

        if self.unsupervised_training_enabled:
            # if we enable unsupervised training, we need to double the batch size for actor model
            # args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
            raise ValueError("Unsupport unsupervised training yet")

        # If passed along, set the training seed now.
        set_random_seed(runtime_conf.seed)
        torch.distributed.barrier()

        # create common tokenizer based on actor model
        if model_provider.tokenizer:
            tokenizer = model_provider.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_provider.initial_model_path,
                                                      fast_tokenizer=True)
            tokenizer.pad_token = tokenizer.eos_token

        if model_conf.self.actor_gen_kwargs or model_conf.actor_eval_gen_kwargs:
            self.actor_gen_kwargs = model_conf.self.actor_gen_kwargs or model_conf.actor_eval_gen_kwargs
        else:
            self.actor_gen_kwargs = {}

        if isinstance(self.train_conf.data_conf.dataset, DataLoader):
            self.prompt_train_dataloader = self.train_conf.data_conf.dataset
            self.unsupervised_train_dataloader = [None] * len(self.prompt_train_dataloader)  # basically a dummy dataloader
        elif inspect.isfunction(self.train_conf.data_conf.dataset):
            # 函数用法参考 alignment.rlhf.data.data_utils.instantiate_dataloader
            self.prompt_train_dataloader = self.train_conf.data_conf.dataset(*self.train_conf.data_conf.dataset_args)
            self.unsupervised_train_dataloader = [None] * len(self.prompt_train_dataloader)  # basically a dummy dataloader
        else:
            data_provider = create_datasets
            if self.train_conf.data_conf.data_range is None:
                self.train_conf.data_conf.data_range = (0, 1.0)
            self.prompt_train_dataloader, self.unsupervised_train_dataloader = \
                data_provider(self.context, tokenizer, train_phase=3, data_conf=self.train_conf.data_conf)

        eval_dataloader = None
        if eval_conf and eval_conf.data_conf:
            eval_data_conf = eval_conf.data_conf
            if isinstance(eval_data_conf.dataset, DataLoader):
                eval_dataloader = eval_data_conf.dataset
            elif inspect.isfunction(eval_data_conf.dataset):
                eval_dataloader = eval_data_conf.dataset(*eval_data_conf.dataset_args)
            else:
                if eval_data_conf.data_range is None:
                    eval_data_conf.data_range = (0, 1.0)
                eval_dataloader, _ = create_datasets(self.context, tokenizer, train_phase=3, data_conf=eval_data_conf)

        num_update_steps_per_epoch = min(len(self.prompt_train_dataloader), len(self.unsupervised_train_dataloader)) * \
                                     (self.train_conf.data_conf.batch_size / self.train_conf.rl_train_batch_size) * \
                                     self.context.self.train_conf.self.rl_train_epochs / \
                                     self.context.self.train_conf.actor_optimizer.gradient_accumulation_steps
        num_total_iters = int(self.context.self.train_conf.self.num_train_epochs * num_update_steps_per_epoch)
        setattr(self.context, "num_self.total_step", num_total_iters)
        logger.info(f"total iteration nums:{num_total_iters}")

        if isinstance(model_provider, ACNoneShareSEPModelProvider):
            self.rlhf_engine = DeepSpeedACNoneShareSEPEngine(
                model_provider=model_provider,
                num_total_iters=num_total_iters,
                context=self.context,
                init_models=False,  # 使用distenv初始化
            )            
        elif isinstance(model_provider, ACNoneShareModelProvider):
            model_placement = ModelPlacement(placement, True, True, True, True)
            self.rlhf_engine = DeepSpeedACNoneShareEngine(
                model_provider=model_provider,
                num_total_iters=num_total_iters,
                context=self.context,
                init_models=False,  # 使用distenv初始化
            )
            rl_trainer = assign_module(self.train_conf.rl_algo, False)
        elif isinstance(model_provider, ACShareModelProvider):
            model_placement = ModelPlacement(placement, True, False, True, True)
            self.rlhf_engine = DeepSpeedACShareEngine(
                model_provider=model_provider,
                num_total_iters=num_total_iters,
                context=self.context,
                init_models=False,  # 使用distenv初始化
            )
            rl_trainer = assign_module(self.train_conf.rl_algo, True)
        else:
            raise ValueError(
                "Unknown ACModelProvider type, which should inherit from ACNoneShare-/ACShareModelProvider")
        if placement.all_models_separate():
            self.dist_rlhf_engine = DistributedDeepSpeedSEPACEngine(self.rlhf_engine, model_placement)
            self.rlhf_engine = self.dist_rlhf_engine            
        elif placement.all_models_flatten():
            self.rlhf_engine.init_all_models()
            self.dist_rlhf_engine = self.rlhf_engine
        else:
            self.dist_rlhf_engine = DistributedDeepSpeedACEngine(self.rlhf_engine, model_placement)
            self.rlhf_engine = self.dist_rlhf_engine

        setattr(self.context, "self.rlhf_engine", self.rlhf_engine)

        if model_conf.rlhf_module is not None:
            rl_trainer = model_conf.rlhf_module

        # restore deepspeed ckpt
        if self.context.runtime_conf.restore_conf:
            load_rlhf_checkpoint(self.context.runtime_conf.restore_conf.restore_path, self.dist_rlhf_engine)

        self.trainer = rl_trainer(self.dist_rlhf_engine, tokenizer, self.context)
        mini_batch_size = self.context.self.train_conf.rl_train_batch_size
        self.rollout_size = self.context.self.train_conf.self.rollout_size

        # the first number is how many experience-batch to generate, the second number is the training batch size,
        # where is the micro-batch size used
        rl_dataset_cls = MiniDataset if _is_on_policy(self.context) else RolloutStorage
        self.exp_mini_dataset = rl_dataset_cls(self.rollout_size, mini_batch_size)
        self.unsup_mini_dataset = rl_dataset_cls(self.rollout_size, mini_batch_size)

        if eval_dataloader:
            from alignment.rlhf.hooks.evaluation_hook import EvaluationHook
            hooks.append(EvaluationHook(self.trainer, eval_dataloader, self.context.eval_conf.eval_at_first,
                                        self.context.eval_conf.eval_every_steps, self.context.eval_conf.steps))

        self.num_train_epochs = self.context.self.train_conf.self.num_train_epochs
        self.rl_train_epochs = self.context.self.train_conf.self.rl_train_epochs
        self.actor_gradient_checkpointing = self.context.runtime_conf.actor_conf.gradient_checkpointing
        self.total_step = 0
        self.experience_step = 0
        self.reach_max_steps = False

        # Train!
        _call_hooks(['on_train_start'])
        _call_hooks(['on_experience_learn_start', 'on_experience_make_start'], experience_step=self.experience_step)

    def run(self):
        if self.placement.all_models_separate():
            self._run_separation()
        else:
            self._run_flatten_or_interleaving()
            

    def _run_separation(self):
        num_rollouts = self.context.train_conf.num_rollouts
        rl_train_epochs = self.context.train_conf.rl_train_epochs
        experience_step = 0

        hooks.append(LoggingHook(self.context.train_conf.log_steps, rl_train_epochs, self.context.train_conf.rollout_size))
        # hooks.append(SavingHook(tokenizer=tokenizer, model_provider=model_conf.model_provider))

        # Train!
        # _call_hooks(['on_train_start'])
        # _call_hooks(['on_experience_learn_start', 'on_experience_make_start'], experience_step=experience_step)
        

        rlhf_sep_config = self.context.runtime_conf.rlhf_sep_config

        BATCHES_PER_ROLLOUT = rlhf_sep_config.batches_per_rollout
        ROLLOUT_BATCH_SIZE = rlhf_sep_config.rollout_batch_size

        """对于训练来说，需要训练60(15*5)条数据，
        每次的micro_batch是2，global_batch_size是12，跑5次train。每个micro_batch是2条
        """
        TRAIN_GS_BATCH_SIZE = rlhf_sep_config.train_global_batch_size

        # TRAIN_GS_BATCH_SIZE包含了DP维度，这里就是每次ROLLOUT需要TRAIN多少step
        BATCHES_PER_TRAIN = int(BATCHES_PER_ROLLOUT * ROLLOUT_BATCH_SIZE // TRAIN_GS_BATCH_SIZE)


        # print_throughput_step3(actor_model, critic_model, e2e_time, gen_exp_time, train_time, is_ac_share, rank=0)
        e2e_time, last_e2e_time = None, None
        for roll_id in range(num_rollouts):
            self.trainer.sync()
            print_rank_0(
                f"Beginning of roll_id {roll_id}/{num_rollouts}, rollout size {self.rollout_size}, Total Generation Batches "
                f"{min(len(self.prompt_train_dataloader), len(self.unsupervised_train_dataloader))}", self.global_rank)

            loader = next_data(self.prompt_train_dataloader)

            for _ in range(BATCHES_PER_ROLLOUT):
                out = self.trainer.generate_experience(loader)
            for train_epoch in range(rl_train_epochs):
                self.trainer.set_epoch(train_epoch)
                for _ in range(BATCHES_PER_TRAIN):
                    actor_loss, critic_loss = self.trainer.train_onestep(train_epoch)

        
            
    def _run_flatten_or_interleaving(self):
        for epoch in range(self.num_train_epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{self.num_train_epochs}, rollout size {self.rollout_size}, Total Generation Batches "
                f"{min(len(self.prompt_train_dataloader), len(self.unsupervised_train_dataloader))}", self.global_rank)

            next_batch_prompt, next_batch_unsupervised = None, None
            self.trainer.eval()
            for step, (batch_prompt,
                       batch_unsupervised) in enumerate(zip(self.prompt_train_dataloader, self.unsupervised_train_dataloader)):
                batch_prompt = to_device(batch_prompt, self.device)
                if step == 0:
                    next_batch_prompt, next_batch_unsupervised = batch_prompt, batch_unsupervised
                    continue
                else:
                    cur_batch_prompt, cur_batch_unsupervised = next_batch_prompt, next_batch_unsupervised
                    next_batch_prompt, next_batch_unsupervised = batch_prompt, batch_unsupervised
                    batch_prompt, batch_unsupervised = cur_batch_prompt, cur_batch_unsupervised

                total_inner_step = 2 if step == len(self.prompt_train_dataloader) - 1 else 1
                for _inner_step in range(total_inner_step):
                    if _inner_step == 1:
                        batch_prompt, batch_unsupervised = next_batch_prompt, next_batch_unsupervised
                        next_batch_prompt, next_batch_unsupervised = None, None
                    _call_hooks(['on_experience_batch_start'], experience_step=self.experience_step)

                    if batch_unsupervised is not None:
                        batch_unsupervised = to_device(batch_unsupervised, self.device)
                        unsup_dataset = self.unsup_mini_dataset.add(batch_unsupervised)
                    else:
                        unsup_dataset = self.unsup_mini_dataset.add(
                            [[None] * self.train_conf.data_conf.batch_size])  # set per_device_train_batch_size to mini_bs
                    prompts = batch_prompt['input_ids'] if 'input_ids' in batch_prompt else batch_prompt['prompt']
                    batch_prompt.pop("input_ids", None)
                    batch_prompt.pop("prompt", None)

                    if "prompt_att_mask" in batch_prompt:
                        # deepspeed chat opt model need it
                        batch_prompt["attention_mask"] = batch_prompt.get("prompt_att_mask")
                        batch_prompt.pop("prompt_att_mask", None)

                    length = prompts.size(-1)
                    if length > self.context.model_conf.max_prompt_seq_len:
                        # prompts = prompts[:, length - self.context.model_conf.max_prompt_seq_len:]
                        raise ValueError("Prompt length is too long")

                    next_prompts = None
                    if next_batch_prompt is not None and isinstance(self.dist_rlhf_engine, DistributedDeepSpeedACEngine):
                        next_prompts = next_batch_prompt['input_ids'] if 'input_ids' in next_batch_prompt else \
                            next_batch_prompt['prompt']
                    batch_prompt['self.experience_step'] = self.experience_step

                    batch_prompt.update(self.actor_gen_kwargs)

                    out = self.trainer.generate_experience(prompts, next_prompts, **batch_prompt)
                    training_start = time.time()

                    if out is None:
                        break

                    # 当add次数达到self.rollout_size时，如果是on policy, exp_dataset返回非None时进入rl训练(ppo)
                    # 如果是off policy, 则在队列满了之后，每次generate_experience都进行一次rl训练
                    exp_dataset = self.exp_mini_dataset.add(out)

                    _call_hooks(['on_experience_batch_end'], experience_step=self.experience_step)

                    # learn

                    if exp_dataset is not None:
                        self.trainer.train()
                        if self.context.runtime_conf.actor_conf.hybrid_engine_conf.release_inference_cache:
                            from deepspeed.runtime.hybrid_engine import inference_cuda_module
                            if inference_cuda_module is not None:
                                inference_cuda_module.release_workspace()
                                gc.collect()
                                torch.cuda.empty_cache()
                        from alignment.rlhf.distributed.distributed_rlhf_engine import SCATTER_QUEUE
                        for i in range(len(SCATTER_QUEUE) - 1, -1, -1):
                            SCATTER_QUEUE[i][0].wait()
                            del SCATTER_QUEUE[i][1]
                            del SCATTER_QUEUE[i]
                        _call_hooks(['on_experience_make_end', 'on_experience_train_start'],
                                    experience_step=self.experience_step)
                        inner_iter = 0
                        critic_loss_sum, actor_loss_sum, unsuper_loss_sum = 0, 0, 0
                        average_reward = 0

                        if self.actor_gradient_checkpointing:
                            if isinstance(self.dist_rlhf_engine, DistributedDeepSpeedACEngine):
                                if self.dist_rlhf_engine.actor.module is not None:
                                    self.dist_rlhf_engine.actor.module.gradient_checkpointing_enable()
                            else:
                                self.dist_rlhf_engine.actor.gradient_checkpointing_enable()

                        for rl_ep in range(self.rl_train_epochs):
                            for i, (exp_data, unsup_data) in enumerate(zip(exp_dataset, unsup_dataset)):
                                _call_hooks('on_experience_train_batch_start', experience_step=self.experience_step,
                                            global_step=self.total_step)

                                next_exp_data = exp_dataset[i + 1] if isinstance(self.dist_rlhf_engine,
                                                                                 DistributedDeepSpeedACEngine) and i < len(
                                    exp_dataset) - 1 else None
                                actor_loss, critic_loss = self.trainer.train_onestep(exp_data, next_exp_data)

                                actor_loss_sum += actor_loss.item()
                                critic_loss_sum += critic_loss.item()
                                cur_reward = exp_data["rewards"].mean()
                                average_reward += cur_reward

                                if self.unsupervised_training_enabled:
                                    unsup_loss = self.trainer.train_unsupervised(unsup_data, self.context.data_conf.unsup_coef)
                                    unsuper_loss_sum += unsup_loss.item()
                                if self.context.model_conf.enable_ema:
                                    moving_average(self.rlhf_engine.actor,
                                                   self.rlhf_engine.actor_ema,
                                                   zero_stage=self.context.runtime_conf.actor_conf.zero_stage)

                                inner_iter += 1
                                self.total_step += 1
                                _call_hooks('on_experience_train_batch_end',
                                            experience_step=self.experience_step,
                                            global_step=self.total_step,
                                            metrics={
                                                'actor_loss': float(actor_loss.item()),
                                                'critic_loss': float(critic_loss.item())
                                            })

                                self.reach_max_steps = 0 < self.context.self.train_conf.max_steps < self.total_step
                                if self.reach_max_steps:
                                    break
                                # end of a batch exp_data of exp_dataset

                            if self.reach_max_steps:
                                break
                            random.shuffle(exp_dataset)
                            random.shuffle(unsup_dataset)
                            # end of a ppo_ep of rl_epochs

                        end = time.time()
                        training_time = end - training_start
                        e2e_time = training_time + self.trainer.generate_time * self.rollout_size  # it is an approximation, we did not include, e.g., rw forward time etc

                        actor_module, critic_module = None, None
                        try:
                            actor_module = self.dist_rlhf_engine.actor.module if isinstance(self.dist_rlhf_engine,
                                                                                       DistributedDeepSpeedACEngine) else self.dist_rlhf_engine.actor
                            critic_module = self.dist_rlhf_engine.critic.module if isinstance(self.dist_rlhf_engine,
                                                                                         DistributedDeepSpeedACEngine) else self.dist_rlhf_engine.critic
                        except:
                            pass

                        is_ac_share = True if isinstance(self.trainer, ACShareDeepSpeedModule) else False
                        print_throughput_step3(actor_module,
                                               critic_module, e2e_time,
                                               self.trainer.generate_time, training_time,
                                               rank=torch.distributed.get_rank(),
                                               is_ac_share=is_ac_share)
                        epoch_metrics = {
                            "epoch": epoch,
                            "step": step,
                            "rl_ep": rl_ep + 1,
                            "avg_act_loss": actor_loss_sum / inner_iter,
                            "avg_cri_loss": critic_loss_sum / inner_iter,
                            "avg_reward": get_all_reduce_mean(average_reward).item() / inner_iter
                        }
                        _call_hooks(['on_experience_train_end'], experience_step=self.experience_step,
                                    global_step=self.total_step,
                                    metrics=epoch_metrics)
                        _call_hooks(['on_experience_learn_end'], experience_step=self.experience_step)

                        if self.reach_max_steps:
                            break

                        self.experience_step += 1
                        _call_hooks(['on_experience_learn_start', 'on_experience_make_start'],
                                    experience_step=self.experience_step)
                        self.trainer.eval()
                    if self.actor_gradient_checkpointing:
                        if isinstance(self.dist_rlhf_engine, DistributedDeepSpeedACEngine):
                            if self.dist_rlhf_engine.actor.module is not None:
                                self.dist_rlhf_engine.actor.module.gradient_checkpointing_disable()
                        else:
                            self.dist_rlhf_engine.actor.gradient_checkpointing_disable()
                            # end of a inner_step of total_inner_step

                if self.reach_max_steps:
                    break
                # end of a batch step of self.prompt_train_dataloader

            if self.reach_max_steps:
                print_rank_0(f"reach max steps {self.context.self.train_conf.max_steps}, end training")
                break
            # end of a epoch of self.num_train_epochs
 
        if self.exp_mini_dataset.has_remaining():
            print_rank_0(f"Skip training remaining data that is less than roll size.")
            self.exp_mini_dataset.free()

        _call_hooks('on_train_end', global_step=self.total_step)