



import os
from typing import Union, Optional, List, Tuple, Any

from alignment.api.rlhf.model_provider import ACModelProvider, ModelProvider
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from alignment.app.util import logger


class GroupPlacePolicy:
    def __init__(self, group_size, owner_id_in_group=0):
        assert owner_id_in_group < group_size
        self._group_size = group_size
        self._dist_groups = None
        self._owner_id_in_group = owner_id_in_group

        self._group_owner_id = None
        self._group_start_id = None

    @property
    def group_size(self):
        return self._group_size

    @property
    def group_start_id(self):
        if self._group_start_id is None:
            import torch
            self._group_start_id = torch.distributed.get_rank() // self.group_size * self.group_size
        return self._group_start_id

    @property
    def group_owner_id(self):
        if self._group_owner_id is None:
            self._group_owner_id = self.group_start_id + self._owner_id_in_group
        return self._group_owner_id

    @property
    def device_id(self):
        # TODO: ensure device id
        import os
        return int(os.getenv("LOCAL_RANK", "0"))

    @property
    def is_owner(self):
        import torch
        return torch.distributed.get_rank() == self.group_owner_id

    @property
    def dist_groups(self):
        if self._dist_groups is None:
            import torch
            assert torch.distributed.get_world_size() % self._group_size == 0
            group_nums = torch.distributed.get_world_size() // self._group_size
            for i in range(group_nums):
                ranks = list(range(i * self._group_size, (i + 1) * self._group_size))
                dist_groups = torch.distributed.new_group(ranks=ranks)
                if self.group_owner_id in ranks:
                    self._dist_groups = dist_groups
            assert self._dist_groups is not None

            print(f'start_rank: {self.group_start_id}, ranks {ranks}, group_size: {self._group_size}'
                  f'group_owner_id: {self.group_owner_id}, device_id {self.device_id}, {self._dist_groups.rank()}')
        return self._dist_groups

class ModelRanks:
    def __init__(self, actor_ranks, critic_ranks, init_model_ranks, reward_model_ranks, pred_actor_ranks,
                 pred_critic_ranks):
        self._actor_ranks = actor_ranks
        self._critic_ranks = critic_ranks
        self._init_model_ranks = init_model_ranks
        self._reward_model_ranks = reward_model_ranks
        self._pred_actor_ranks = pred_actor_ranks
        self._pred_critic_ranks = pred_critic_ranks

    @property
    def actor_ranks(self):
        return self._actor_ranks

    @property
    def critic_ranks(self):
        return self._critic_ranks

    @property
    def init_model_ranks(self):
        return self._init_model_ranks

    @property
    def reward_model_ranks(self):
        return self._reward_model_ranks

    @property
    def pred_actor_ranks(self):
        return self._pred_actor_ranks

    @property
    def pred_critic_ranks(self):
        return self._pred_critic_ranks


class DataConfig:
    def __init__(self,
                 batch_size: int,
                 dataset: Union[str, DataLoader, Any],
                 dataset_args: Tuple = None,
                 data_range: Tuple[float, float] = None,
                 num_workers: int = 1,
                 workdir: str = '/tmp/data',
                 unsupervised_dataset_name=None,
                 unsupervised_dataset_config_name=None,
                 unsup_coef: float = 27.8,
                 ):
        """ Data config.

        Args:
            dataset (str): dataset path, it could be:

                * a name (or a name list) ref to the dataset(s) in hugging face, such as 'Dahoas/rm-static',
                    or ['Dahoas/rm-static', 'Dahoas/rl-prompt-dataset']

                * a local path (or a list) to the dataset.

                * an instance of DataLoader, may not supported in dist env.

                * a function returning a DataLoader, with args in dataset_args.
            dataset_args (Tuple): Args for callable dataset. If the dataset is an instance of DataLoader,
                this field takes no effect. The shuffle option can be set in DataLoader directly.
            data_range (List[float]): Read records between ``data_range * table_size``.
            batch_size(int): The batch size of the dataloader in rollout phase.
            num_workers(int): Preprocessing worker num, default to 1.
        """
        pass


class LoraConfig:
    def __init__(self,
                 actor_lora_dim: int = 128,  # alias lora_dim
                 critic_lora_dim: Optional[int] = 1,
                 actor_lora_module_name='.layers.',  # alias lora_module_name
                 critic_lora_module_name='.layers.',
                 only_optimize_lora=True,
                 **kwargs
                 ):
        """LoRA config
        Args:
            actor_lora_dim(int): lora dim for A's output and B's input.
            actor_lora_module_name(str): The module whose name matches this will be convert to lora layer.
                Note: the module name is specific with LLM.
                For OPT, module_name can be 'decoder.layers.';
                For ChatGLM, module_name can be '.transformer.layers.';
                For GLM, module_name can be 'transformer.layers.';
                For GPT-j, lora-config is not recommended, for no nn.Linear layers in those block
        """
        pass

    # alias for lora_dim
    @property
    def lora_dim(self):
        return self.actor_lora_dim

    @lora_dim.setter
    def lora_dim(self, lora_dim):
        self.actor_lora_dim = lora_dim

    @property
    def lora_module_name(self):
        return self.actor_lora_module_name

    @lora_module_name.setter
    def lora_module_name(self, lora_module_name):
        self.actor_lora_module_name = lora_module_name


class FrozenLayerConfig:
    def __init__(self,
                 actor_num_layers_unfrozen: int = 8,
                 critic_num_layers_unfrozen: int = 8
                 ):
        pass


class ModelConfig:

    def __init__(self,
                 model_provider: Union[ACModelProvider, ModelProvider],
                 rlhf_module=None,
                 num_padding_at_beginning=1,
                 max_prompt_seq_len=256,
                 max_answer_seq_len=256,
                 end_of_conversation_token="<|endoftext|>",
                 model_arch_type='causal',
                 enable_ema=False,
                 enable_ref=True,
                 update_ref=False,
                 update_ref_per_step=1,
                 need_value_head=False,
                 lora_conf: Optional[LoraConfig] = None,
                 frozen_layer_conf: Optional[FrozenLayerConfig] = None,
                 vf_coef=1.0,
                 rl_kl_ctl=0.02,
                 rl_clip_reward_value=0.2,
                 rl_cliprange=0.2,
                 rl_cliprange_value=0.2,
                 rl_gamma=1.0,
                 rl_lam=0.95,
                 forward_use_cache=False,
                 run_ref_reward_async=False,
                 actor_gen_kwargs=None,
                 actor_eval_gen_kwargs=None,
                 input_gen_func=None,
                 ):
        """Model config.

        Args:
            model_provider (ACModelProvider): An instance of ACModelProvider, which carrying
                init_model_path and reward_model_path.
            num_padding_at_beginning(int): The number of padding at beginning.
            model_arch_type(str): The model arch type, causal as default. (seq2seq will come soon)
            enable_ema (bool): Whether or not enable ema(Exponential moving average) for parameters.
                If True, will add another model to collect the expotential moving average of the actor model's weight.
                According to InstructGPT, the EMA weight has better performance than actor model's final checkpoint.
            enable_ref (bool): Whether enable ref model or not, default True. If set False, ref model won't be created.
            update_ref (bool): Whether updating the ref model or not, default False.
            update_ref_per_step (int): If update_ref is True, then update ref per update_ref_per_step.
            need_value_head (bool): Whether or not need to add a value head to the actor model. If the model with added
                head is available in model_provider, you can set it to False.
            lora_conf (Optional[LoraConfig]): Config for LoRA, exclusive with frozen_layer_conf.
            frozen_layer_conf (Optional[FrozenLayerConfig]): Config for layer frozen, exclusive with lora_conf.

            vf_coef (float): The value function coefficient for the loss calculation.
            rl_kl_ctl(float): The KL-divergence control vale.
            rl_clip_reward_value(float): The clip value for reward score.
            rl_cliprange(float): The clip range for actor in PPO.
            rl_cliprange_value(float): The clip range for critic in PPO.
            rl_gamma(float): RL gamma, which is used in computing advantages.
            rl_lam(float): RL lam, which is used in computing advantages.
        """
        if frozen_layer_conf and run_ref_reward_async:
            self.run_ref_reward_async = run_ref_reward_async = False
            logger.warning('Both run_ref_reward_async and frozen_layer_conf are set, reset the run_ref_reward_async to False')
        if lora_conf and frozen_layer_conf:
            self.frozen_config = None
            logger.warning('Both lora_config and frozen_layer_conf are set, reset the frozen_layer_conf to None')
        if lora_conf is None and frozen_layer_conf is None:
            logger.warning('Neither lora_conf nor frozen_layer_conf are set, '
                           'the whole LM will be trained, which is time consuming!')



class SchedulerConfig:

    def __init__(self,
                 lr_scheduler_type='cosine',
                 num_warmup_steps=100):
        """Scheduler config.

        Args:
            lr_scheduler_type(str): Learning rate scheduler type.
            num_warmup_steps(int): The num of warmup steps.
        """
        pass



class OptimizerConfig:

    def __init__(self,
                 name: str = None,
                 instance: Optional[Optimizer] = None,
                 learning_rate=9.65e-6,
                 weight_decay=0.1,
                 gradient_accumulation_steps=1,
                 scheduler_conf: Optional[SchedulerConfig] = None,
                 ):
        """Optimizer config.

        Args:
            name: The optimizer name.
            instance: An instance of torch.optim.optimizer.Optimizer. If none, will auto set according to
                distributed config. If set, no need to set learning_rate, weight_decay.
        """
        pass



class EngineDistributedConfig:
    def __init__(self,
                 enable_hybrid_engine=True,
                 unpin_actor_parameters=False,
                 release_inference_cache=True,
                 inference_tp_size=1,
                 tp_gather_partition_size=8,
                 ):
        """Distributed Engine Config.

        Args:
            enable_hybrid_engine(bool): Enable it to use DeepSpeed Hybrid Engine. This will significantly speedup your
                RL training.
            unpin_actor_parameters(bool): Do not gather the actor parameter for generation(if set True). This will
                significantly slow down the generation phase. Usually set False.
            release_inference_cache(bool): Release the memory reserved for sentence generation. This will slow down
                the training a bit but perhaps increasing the training batch size.
            inference_tp_size(int): The inference tensor parallelism size, do not exceed the size of a single node.
            tp_gather_partition_size(int): Granularity to bring in layers for TP sharding inside the hybrid engine.
                Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature.
        """
        pass

    @property
    def unpin_parameters(self):
        return self.unpin_actor_parameters

    @unpin_parameters.setter
    def unpin_parameters(self, unpin_parameters):
        self.unpin_actor_parameters = unpin_parameters



class DistributedConfig:

    def __init__(self,
                 zero_stage=0,
                 gradient_checkpointing=True,
                 offload=True,
                 steps_per_print=10,
                 hybrid_engine_conf: EngineDistributedConfig = None,
                 deepspeed_extra_kwargs=None,
                 enable_init_partition=False,
                 ):
        """Deep speed distributed Config in code for role.

        Args:
            zero_stage:ZeRO optimization stage, 0 means disabled.
                If the model is actor, its clones will share the same config.
            deepspeed_extra_kwargs: The extra kwargs of deepSpeed, in dict format.
        """
        if enable_init_partition and zero_stage != 3:
            self.enable_init_partition = enable_init_partition = False



class RestoreConfig:
    ACTOR_TAG = "actor"
    CRITIC_TAG = "critic"
    SFT_TAG = "sft"

    def __init__(self,
                 restore_path: str,
                 restore_io_state=False
                 ):
        """Restore actor or critic from restore path.

        Args:
            restore_path(str): The path to the checkpoint. If actor-critic is not sharing, both of "actor" and "critic"
                sub-folder are needed in the restore_path. If shared, the "actor" folder is needed.
            restore_io_state(bool): Restore IO state. (Not implemented yet, default to False)
        """
        pass


class ModelSEPConfig(object):
    """Configure for Profiling"""

    def __init__(self,
                 megatron_args: dict,
                 TP: int = 8,
                 PP: int = 1,
                 replicas: int = 1,
                 checkpoint_path: str=None,
                 checkpointing: bool=False,
                 sequence_parallel: bool=False,
                 ):
        """Initialize ProfilerConfig
        目前仅train_model支持PP
        replicas仅pred_actor支持
        """
        megatron_args['tensor_model_parallel_size'] = TP
        megatron_args['pipeline_model_parallel_size'] = PP
        if checkpoint_path is not None:
            megatron_args['load'] = checkpoint_path
            megatron_args['load_iteration'] = -1

        if checkpointing:
            megatron_args['recompute_activations'] = True
            megatron_args['recompute_granularity'] = 'full'
        self.num_layers = megatron_args['num_layers']
        if sequence_parallel:
            megatron_args['sequence_parallel'] = True



class RLHFSEPConfig(object):
    """Configure for Profiling"""

    def __init__(self,
                 actor_sep_config: ModelSEPConfig,
                 critic_sep_config: ModelSEPConfig,
                 pred_actor_sep_config: ModelSEPConfig,
                 pred_critic_sep_config: ModelSEPConfig,
                 initial_sep_config: ModelSEPConfig,
                 reward_sep_config: ModelSEPConfig,
                 rollout_batch_size: int,
                 train_micro_batch_size: int,
                 batches_per_rollout: int,
                 train_global_batch_size: int,
                 tokenizer_model: str,
                 max_new_tokens: int,
                 seq_length: int,
                 input_length: int,
                 bf16: bool=True,
                 ):
        """目前先假定actor/critic的train_batch_size相等
        """
        assert actor_sep_config.num_layers == pred_actor_sep_config.num_layers == initial_sep_config.num_layers
        assert critic_sep_config.num_layers == pred_critic_sep_config.num_layers == reward_sep_config.num_layers

        # 包含了数据并行
        actor_sep_config.megatron_args['global_batch_size'] = critic_sep_config.megatron_args['global_batch_size'] = train_global_batch_size
        actor_sep_config.megatron_args['micro_batch_size'] = critic_sep_config.megatron_args['micro_batch_size'] = train_micro_batch_size

        # pred无数据并行，micro_batch_size * 1 == global_batch_size
        pred_actor_sep_config.megatron_args['micro_batch_size'] = pred_critic_sep_config.megatron_args['micro_batch_size'] = rollout_batch_size
        initial_sep_config.megatron_args['micro_batch_size'] = reward_sep_config.megatron_args['micro_batch_size'] = rollout_batch_size

        initial_sep_config.megatron_args['tokenizer_model'] = reward_sep_config.megatron_args['tokenizer_model'] = tokenizer_model
        pred_actor_sep_config.megatron_args['tokenizer_model'] = pred_critic_sep_config.megatron_args['tokenizer_model'] = tokenizer_model
        actor_sep_config.megatron_args['tokenizer_model'] = critic_sep_config.megatron_args['tokenizer_model'] = tokenizer_model

        pred_actor_sep_config.megatron_args['max_new_tokens'] = max_new_tokens


        initial_sep_config.megatron_args['seq_length'] = reward_sep_config.megatron_args['seq_length'] = seq_length
        pred_actor_sep_config.megatron_args['seq_length'] = pred_critic_sep_config.megatron_args['seq_length'] = seq_length
        actor_sep_config.megatron_args['seq_length'] = critic_sep_config.megatron_args['seq_length'] = seq_length

        initial_sep_config.megatron_args['out_seq_length'] = reward_sep_config.megatron_args['out_seq_length'] = seq_length
        pred_actor_sep_config.megatron_args['out_seq_length'] = pred_critic_sep_config.megatron_args['out_seq_length'] = seq_length
        actor_sep_config.megatron_args['out_seq_length'] = critic_sep_config.megatron_args['out_seq_length'] = seq_length


        initial_sep_config.megatron_args['bf16'] = reward_sep_config.megatron_args['bf16'] = bf16
        pred_actor_sep_config.megatron_args['bf16'] = pred_critic_sep_config.megatron_args['bf16'] = bf16
        actor_sep_config.megatron_args['bf16'] = critic_sep_config.megatron_args['bf16'] = bf16


class RuntimeConfig:

    def __init__(self,
                 actor_conf_path: Optional[str] = None,
                 critic_conf_path: Optional[str] = None,
                 actor_conf: Optional[DistributedConfig] = None,
                 critic_conf: Optional[DistributedConfig] = None,
                 offload_reference_model: bool = True,
                 seed=None,
                 reward_policy: Optional[GroupPlacePolicy] = None,
                 ref_policy: Optional[GroupPlacePolicy] = None,
                 target_policy: Optional[GroupPlacePolicy] = None,
                 custom_port: Union[bool, int] = False,
                 restore_conf: Optional[RestoreConfig] = None,
                 model_dir: Optional[str] = './base_model',
                 log_dir: Optional[str] = None,
                 summary_every_step: int = 100,
                 dtype="bf16",
                 zero_opt_param: Optional[dict] =None,
                 skip_load: [bool] = False,
                 rlhf_sep_config: RLHFSEPConfig = None,
                 **kwargs
                 ):
        """Runtime config.

        Args:
            actor_conf_path(str): Distribute config path for actor. The config file is prior to the code.
            actor_conf(DistributedConfig): Distribute config for actor, exclusive with actor_conf_path.
            offload_reference_model(bool): Whether or not offload reference model to CPU.
                This helps increase the batch size with negligible time cost.
            seed: The random seed for reproducibility.
            target_policy: The optional GroupPlacePolicy for target model (the critic in ac share mode).
            custom_port(Union[bool, int]): Set True, will auto generate a dynamic port.
                Set to an integer, will use the port directly. Above cases support running more than one job in single
                machine. Default is False.
            model_dir(str): The model dir to save events for dashboard, or to save checkpoint.
            summary_every_step(int): log summary info every N step
        """
        if log_dir is None and model_dir is not None:
            self.log_dir = os.path.join(model_dir, "logs")

    # alias
    @property
    def dist_conf(self):
        # distributed conf
        return self.actor_conf

    @dist_conf.setter
    def dist_conf(self, conf):
        self.actor_conf = conf


class TrainConfig:

    """Configuration for the training process."""

    def __init__(self,
                 data_conf: DataConfig = None,
                 rl_algo: str = "PPO2",
                 rl_policy: str = "on",
                 num_train_epochs: Optional[int] = 1,
                 rl_train_epochs: Optional[int] = 1,
                 rl_train_batch_size: Optional[int] = None,
                 rollout_size=16,
                 max_steps: Optional[int] = -1,
                 log_steps: Optional[int] = 100,
                 actor_optimizer: OptimizerConfig = None,
                 critic_optimizer: Optional[OptimizerConfig] = None,
                 user_defined_hooks: List[ConfigurableHook] = None,
                 save_checkpoints_steps: Optional[int] = 5000,
                 keep_checkpoint_max: Optional[int] = 2,
                 checkpoint_dir: Optional[str] = None,
                 **kwargs
                 ):
        """Train config for RLHF.

        Args:
            data_conf(DataConfig): Data config for train.
            rl_algo(str): RL algorithm. Options: PPO, PPO2. Coming soon: GDC, SAC, TD3, BC, CQL, IQL.
            rl_policy(str): RL policy. Options: on or off, for on-policy or off-policy respectively, default on.
            actor_optimizer(OptimizerConfig): Optimizer config (for actor in RLHF).
            critic_optimizer(Optional[OptimizerConfig]): Optimizer config for critic in RLHF no-share mode.
            num_train_epochs(int): The num of (outer iter) train epoch for each trajectory (namely experience).
            rl_train_epochs(int): The num of (inner iter) train epoch for Reinforcement Learning algorithm (such as PPO).
            rl_train_batch_size(int): The batch size used in rl train. If None, will be set same value as the batch_size
                in DataConfig.
                rl_train_dataset_num = rollout_num = batch_size (in DataConfig) * rollout_size
                rl_train_total_step = rl_train_epochs * rollout_num / rl_train_batch_size
            rollout_size(int): The rollout size in each rl_train_epoch.
            max_steps(int): The max (rl train step) steps to train. If <=0, will end by rl_train_epochs.
            user_defined_hooks: User defined hooks. (Coming soon)
            save_checkpoints_steps: save checkpoints every steps.
            keep_checkpoint_max(int): The max number of keeping checkpoint.
        """
        if max_steps is None:
            self.max_steps = -1

    # alias for actor_optimizer
    @property
    def optimizer(self):
        return self.actor_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.actor_optimizer = optimizer



class EvalConfig:

    def __init__(self,
                 data_conf: DataConfig = None,
                 eval_every_steps: Optional[int] = None,
                 eval_at_first: Optional[bool] = False,
                 steps: int = None,
                 compute_metric_fn=None
                 ):
        """Initialize EvalConf

        Args:
            data_conf (DataConfig): Data config for eval.
            eval_every_steps (int): If set, trigger parallel validation every this many steps instead of
                by saving checkpoint.
            eval_at_first (bool): Do evaluation at first.
            compute_metric_fn: The function for computing metrics when evaluating. For example:
                >>> def compute_metric_fn(dataloader, model, device, max_steps):
                >>>     for step_idx, batch in tqdm(enumerate(dataloader)):
                >>>         with torch.no_grad():
                >>>             outputs = model(**batch)
                >>>
                >>>         loss = outputs.loss
                >>>         losses += loss.float()
                >>>         if max_steps and step_idx > max_steps:
                >>>             break
                >>>     losses = losses / (step_idx + 1)
                >>>     return dict(loss=losses)
                You can pass compute_metrics of HF Trainer.
        """
        pass



class Slot:

    def __init__(self, actor_ratio=1.0,
                 critic_ratio=1.0,
                 initial_ratio=1.0,
                 reward_ratio=1.0):
        pass



class Placement:
    def __init__(self, slot=Slot(), model_ranks=None):
        pass

    def all_models_flatten(self):
        return False

    def all_models_separate(self):
        return False


class InitialRewardSeparate(Placement):

    def __init__(self):
        slot = Slot(1, 1, 0.5, 0.5)

        super(InitialRewardSeparate, self).__init__(slot)


class AllCollocate(Placement):

    def __init__(self):
        slot = Slot(1, 1, 1, 1)
        super(AllCollocate, self).__init__(slot)

    def all_models_flatten(self):
        return True


class AllSeparate(Placement):

    def __init__(self, model_ranks):
        super(AllSeparate, self).__init__(model_ranks=model_ranks)
    
    def all_models_separate(self):
        return True



class ProfilerConfig(object):
    """Configure for Profiling"""

    def __init__(self,
                 save_steps: int = 10,
                 max_profile_cnt: int = 5,
                 skip_first_steps: int = 10,
                 enable_flops_profile: bool = True,
                 #  rank0_only: bool = True,
                 enable_torch_profile: bool = False):
        """Initialize ProfilerConfig

        Args:
        """
        pass