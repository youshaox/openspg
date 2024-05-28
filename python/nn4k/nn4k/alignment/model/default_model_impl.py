# coding: utf-8

from alignment.api.rlhf.model_provider import ACShareModelProvider, ACNoneShareModelProvider, ModelProvider

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from alignment.rlhf.model.model_utils import create_hf_model, create_critic_model, create_reward_model


class DefaultACNoneShareModelProvider(ACNoneShareModelProvider):

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path=None,
                 tokenizer=None, num_padding_at_beginning=1, **kwargs):
        print("DefaultChatModelProvider init")
        super(DefaultACNoneShareModelProvider, self).__int__(actor_model_name_or_path, critic_model_name_or_path)
        self.actor_model_name_or_path = actor_model_name_or_path
        self.critic_model_name_or_path = critic_model_name_or_path
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(actor_model_name_or_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.num_padding_at_beginning = num_padding_at_beginning
        self.kwargs = kwargs

    def get_actor(self, ds_config):

        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=self.actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            disable_dropout=self.kwargs.get("disable_actor_dropout", True))
        return actor_model

    def get_critic(self, ds_config):
        # Model
        critic_model = create_critic_model(
            model_name_or_path=self.critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            num_padding_at_beginning=self.num_padding_at_beginning,
            rlhf_training=True,
            disable_dropout=self.kwargs.get("disable_critic_dropout", False),
        )
        return critic_model

    def get_reward_model(self, ds_config):
        reward_model = create_critic_model(
            model_name_or_path=self.critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            num_padding_at_beginning=self.num_padding_at_beginning,
            rlhf_training=True)
        return reward_model

    def get_initial_model(self, ds_config):
        ref_model = create_hf_model(
            AutoModelForCausalLM,
            self.actor_model_name_or_path,
            self.tokenizer,
            ds_config)
        return ref_model

    def get_ema(self, ds_config):
        actor_model_ema = create_hf_model(
            AutoModelForCausalLM,
            self.actor_model_name_or_path,
            self.tokenizer, ds_config)
        return actor_model_ema


class DefaultACShareModelProvider(ACShareModelProvider):
    def __init__(self, actor_model_name_or_path, critic_model_name_or_path=None,
                 tokenizer=None, num_padding_at_beginning=1, **kwargs):
        super(DefaultACShareModelProvider, self).__int__(actor_model_name_or_path, critic_model_name_or_path)
        self.actor_model_name_or_path = actor_model_name_or_path  # namely sft
        self.critic_model_name_or_path = critic_model_name_or_path  # namely reward
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(actor_model_name_or_path)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.num_padding_at_beginning = num_padding_at_beginning
        self.kwargs = kwargs

    def get_actor(self, ds_config):
        actor = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=self.actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config)
        return actor

    def get_reward_model(self, ds_config):
        reward_model = create_reward_model(
            model_name_or_path=self.critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            num_padding_at_beginning=self.num_padding_at_beginning,
            rlhf_training=True)
        return reward_model


class DefaultModelProvider(ModelProvider):

    def _wrap_lora(self, model, model_conf):
        lora_conf = model_conf.lora_conf
        if lora_conf and lora_conf.actor_lora_dim > 0:
            # LoRA 实现
            from alignment.rlhf.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible

            model = convert_linear_layer_to_lora(model, lora_conf.lora_module_name, lora_conf.lora_dim)
            if lora_conf.only_optimize_lora:
                model = only_optimize_lora_parameters(model)
                model = make_model_gradient_checkpointing_compatible(model)
        return model

    def get_model(self, model_conf, ds_config):
        model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=self.model_path,
            tokenizer=self.tokenizer,
            rlhf_training=True,
            ds_config=ds_config)

        model = self._wrap_lora(model, model_conf)

        return model
