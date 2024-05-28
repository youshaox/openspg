# coding: utf-8

from abc import abstractmethod
from alignment.rlhf.model.model_decoration import ModelDecoration


@abstractmethod
class ModelProvider(object):

    def __init__(self,
                 model_path: str,
                 tokenizer=None,
                 **kwargs):
        self.model_path = model_path
        self.tokenizer = tokenizer

    def get_model(self, **kwargs):
        return None


@abstractmethod
class ACModelProvider(object):

    def __int__(self,
                initial_model_path: str,
                reward_model_path: str,
                tokenizer=None,
                **kwargs):
        self.initial_model_path = initial_model_path
        self.reward_model_path = reward_model_path
        self.tokenizer = tokenizer

    def get_reward_model(self, **kwargs):
        pass

    def get_initial_model(self, **kwargs):
        return None


class ACNoneShareModelProvider(ACModelProvider, ModelDecoration):

    def get_actor_model(self, model_config, **kwargs):
        actor = self.get_actor(**kwargs)
        return self.decorate_actor(actor, model_config)

    def get_critic_model(self, model_conf, **kwargs):
        critic = self.get_critic(**kwargs)
        return self.decorate_critic(critic, model_conf)

    def get_ema_model(self, model_conf, **kwargs):
        ema = self.get_ema(**kwargs)
        return self.decorate_ema(ema, model_conf)

    def get_actor(self, **kwargs):
        pass

    def get_critic(self, **kwargs):
        pass

    def get_ema(self, **kwargs):
        pass

class ACNoneShareSEPModelProvider(ACNoneShareModelProvider):

    def get_actor_model(self, model_config, **kwargs):
        return self.get_actor(**kwargs)

    def get_critic_model(self, model_conf, **kwargs):
        return self.get_critic(**kwargs)
    
    def get_pred_actor(self, **kwargs):
        pass

    def get_pred_critic(self, **kwargs):
        pass

    def get_pred_actor_model(self, model_config, **kwargs):
        return self.get_pred_actor(**kwargs)

    def get_pred_critic_model(self, model_conf, **kwargs):
        return self.get_pred_critic(**kwargs)


class ACShareModelProvider(ACModelProvider, ModelDecoration):

    def get_actor_model(self, model_config, **kwargs):
        actor = self.get_actor(**kwargs)
        return self.decorate_actor(actor, model_config)

    def get_actor(self, **kwargs):
        """返回原始的hf模型，无需指定cuda()/half()，这些操作会在deepspeed里实现"""
        pass
