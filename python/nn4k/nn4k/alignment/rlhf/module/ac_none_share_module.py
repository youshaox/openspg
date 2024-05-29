# coding: utf-8
import time

import torch

from alignment.rlhf.module.rlhf_module import DeepSpeedRLHFModule, gather_log_probs
from alignment.rlhf.distributed.distributed_rlhf_engine import TrainModel
from alignment.rlhf.distributed.distributed_rlhf_engine import DistributedDeepSpeedACEngine
from alignment.app.util import logger
from alignment.rlhf.utils.perf import print_throughput_step3_sep

MAPPING_RANKS = dict()


class ACNoneShareDeepSpeedModule(DeepSpeedRLHFModule):
    def __init__(self, rlhf_engine, tokenizer, context):
        super(ACNoneShareDeepSpeedModule, self).__init__(rlhf_engine, tokenizer, context)
        self.critic_model = self.rlhf_engine.critic

    def generate_experience(self, prompts, next_prompts=None, **kwargs):
        self.eval()
        experience_step = kwargs.pop('experience_step', 1)

        generate_start = time.time()
        # if torch.distributed.get_rank() == 0:
        #     import pdb; pdb.set_trace()
        # 使用loader里的mask
        attention_mask = kwargs.get('attention_mask', None)
        seq, kwargs = self._generate_sequence(prompts, next_prompts, kwargs, mask=attention_mask)

        # if torch.distributed.get_rank() == 0:
        #     import pdb; pdb.set_trace()
        # time.sleep(10000)        
        self.generate_time = time.time() - generate_start
        if seq is None:
            return seq

        from alignment.rlhf.trainner.app_ds_rlhf_trainner import _call_hooks
        _call_hooks(['on_experience_batch_start'], experience_step=experience_step)

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        output, output_ref, reward_score, values = self._generate_experience_by_seq(attention_mask, seq)
        ## scatter ......

        logits = output.logits

        self.train()

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': output_ref,
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }

    def _generate_experience_by_seq(self, attention_mask, seq):
        forward_use_cache = self.context.model_conf.forward_use_cache
        run_ref_reward_async = False
        with torch.no_grad():
            if isinstance(self.rlhf_engine,
                          DistributedDeepSpeedACEngine) and self.context.model_conf.run_ref_reward_async:
                run_ref_reward_async = True


            # logger.info(f'Before run actor {torch.cuda.memory_allocated() / 1e9} {torch.cuda.max_memory_allocated() / 1e9}') 
            output = self.actor_model(seq, attention_mask=attention_mask, return_dict=True, use_cache=forward_use_cache)
            # output.logits = gather_log_probs(output.logits[:, :-1, :], seq[:, 1:])
            # if torch.distributed.get_rank() == 0:                                      
            #     logger.info(f'After run actor {torch.cuda.memory_allocated() / 1e9} {torch.cuda.max_memory_allocated() / 1e9}')                                     


            # if torch.distributed.get_rank() == 0:
            #     import pdb
            #     pdb.set_trace()
            if self.ref_model.module is not None or not run_ref_reward_async:
                # costly [bs, seq_len, vocab_size], 及时gather清理掉
                ref_res = self.ref_model(seq, attention_mask=attention_mask, use_cache=forward_use_cache, return_dict=True)
            #     ref_res.logits = gather_log_probs(ref_logits[:, :-1, :], seq[:, 1:])                                         
            # if torch.distributed.get_rank() == 0:                                         
            #     logger.info(f'After run ref {torch.cuda.memory_allocated() / 1e9} {torch.cuda.max_memory_allocated() / 1e9}')                                                                              

            if self.reward_model.module is not None or not run_ref_reward_async:
                # [bs, seq_len, vocab_size], 及时gather清理掉
                reward_res = self.reward_model.forward_value(seq,
                                                             attention_mask,
                                                             prompt_length=self.prompt_length,
                                                             use_cache=forward_use_cache)
            # if torch.distributed.get_rank() == 0:                                                             
            #     logger.info(f'After run reward {torch.cuda.memory_allocated() / 1e9} {torch.cuda.max_memory_allocated() / 1e9}')                                                                                                                                           
            values = self.critic_model.forward_value(seq,
                                                     attention_mask,
                                                     return_value_only=True,
                                                     use_cache=forward_use_cache).detach()[:, :-1]
            # if torch.distributed.get_rank() == 0:                                                     
            #     logger.info(f'After run critic {torch.cuda.memory_allocated() / 1e9} {torch.cuda.max_memory_allocated() / 1e9}')                                                                                                                                   
            # if torch.distributed.get_rank() == 0:
            #     import pdb
            #     pdb.set_trace()                                                     
            if run_ref_reward_async:
                bs, seq_len = seq.shape[0:2]
                seq_len -= 1

                if self.ref_model.module is not None:
                    prev_idx, cur_len = 0, bs * (seq_len + 1)
                    output_tensor = self.ref_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view(-1)

                    prev_idx += cur_len
                    cur_len = bs * seq_len * 2
                    input_tensor = self.ref_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view((bs * 2, seq_len))
                    input_tensor = torch.cat(ref_res, out=input_tensor).view(-1)

                    del ref_res
                    self.ref_model._all_to_all_single(input_tensor,
                                                      output_tensor,
                                                      input_split_sizes=[bs * seq_len, bs * seq_len],
                                                      output_split_sizes=[bs * seq_len, bs],
                                                      run_async=True)

                if self.reward_model.module is not None:
                    prev_idx, cur_len = 0, bs * (seq_len + 1)
                    output_tensor = self.reward_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view(-1)

                    prev_idx += cur_len
                    cur_len = bs * 2
                    input_tensor = self.reward_model.tmp_empty_tensor.narrow(0, prev_idx, cur_len).view((bs * 2))
                    input_tensor = torch.cat(reward_res, out=input_tensor).view(-1)
                    del reward_res

                    self.reward_model._all_to_all_single(input_tensor,
                                                         output_tensor,
                                                         input_split_sizes=[bs, bs],
                                                         output_split_sizes=[bs * seq_len, bs],
                                                         run_async=True)

                reward_score = output_tensor[bs * seq_len:].view((bs))
                ref_logits = output_tensor[0:bs * seq_len].view((bs, seq_len))

            else:
                ref_logits = ref_res.logits
                reward_score = reward_res['chosen_end_scores'].detach()
            if len(ref_logits.shape) == 2:
                ref_logprobs = ref_logits
            else:
                ref_logprobs = gather_log_probs(ref_logits[:, :-1, :], seq[:, 1:])
        return output, ref_logprobs, reward_score, values

    def train_onestep(self, inputs, next_inputs):
        from alignment.rlhf.hooks.profile_train_hook import TraceEventScope

        @TraceEventScope("train_actor")
        @TrainModel(self.actor_model, self.rlhf_engine, True)
        def train_actor(inputs, next_inputs):
            # train the rlhf mode here
            ### process the old outputs
            prompts = inputs['prompts']
            log_probs = inputs['logprobs']
            ref_log_probs = inputs['ref_logprobs']
            reward_score = inputs['rewards']
            values = inputs['value']
            attention_mask = inputs['attention_mask']
            seq = inputs['input_ids']

            start = prompts.size()[-1] - 1
            action_mask = attention_mask[:, 1:]

            old_values = values
            with torch.no_grad():
                old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
                ends = start + action_mask[:, start:].sum(1) + 1
                # we need to zero out the reward and value after the end of the conversation
                # otherwise the advantage/return will be wrong
                for i in range(old_rewards.shape[0]):
                    old_rewards[i, ends[i]:] = 0
                    old_values[i, ends[i]:] = 0
                advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

            logger.info('summation of old_rewards: {}, advantages:{}, returns: {}'.format(torch.sum(old_rewards), torch.sum(advantages), torch.sum(returns)))
            logger.info('shape of old_rewards: {}, advantages:{}, returns: {}'.format(old_rewards.shape, advantages.shape, returns.shape))

            ### process the new outputs
            batch = {'input_ids': seq, "attention_mask": attention_mask}
            actor_prob = self.actor_model(**batch, use_cache=False, return_dict=True).logits
            actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], inputs['input_ids'][:, 1:])
            
            actor_loss = self._actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, action_mask[:, start:])                    
            self.actor_model.backward(actor_loss)
            logger.info(f'actor_loss: {actor_loss}')
            return actor_loss

        @TraceEventScope("train_critic")
        @TrainModel(self.critic_model, self.rlhf_engine, True)
        def train_critic(inputs, next_inputs):
            # critic和actor可能device rank不重合，这里进行分拆。部分前向数据后续也可复用？
            prompts = inputs['prompts']
            log_probs = inputs['logprobs']
            ref_log_probs = inputs['ref_logprobs']
            reward_score = inputs['rewards']
            values = inputs['value']
            attention_mask = inputs['attention_mask']
            seq = inputs['input_ids']

            start = prompts.size()[-1] - 1
            action_mask = attention_mask[:, 1:]
            ### process the new outputs
            batch = {'input_ids': seq, "attention_mask": attention_mask}
            old_values = values
            with torch.no_grad():
                old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
                ends = start + action_mask[:, start:].sum(1) + 1
                # we need to zero out the reward and value after the end of the conversation
                # otherwise the advantage/return will be wrong
                for i in range(old_rewards.shape[0]):
                    old_rewards[i, ends[i]:] = 0
                    old_values[i, ends[i]:] = 0
                advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)
            value = self.critic_model.forward_value(**batch, return_value_only=True, use_cache=False)[:, :-1]
            critic_loss = self._critic_loss_fn(value[:, start:], old_values[:, start:], returns,
                                               action_mask[:, start:])
            self.critic_model.backward(critic_loss)
            logger.info(f'critic_loss: {critic_loss}')            
            return critic_loss

        actor_loss = train_actor(inputs, next_inputs)
        critic_loss = train_critic(inputs, next_inputs)

        self.actor_model.step()
        self.critic_model.step()
        logger.info(f'actor_loss: {actor_loss}, critic_loss: {critic_loss}')        

        return actor_loss, critic_loss

class ACNoneShareDeepSpeedSEPModule(DeepSpeedRLHFModule):
    def __init__(self, rlhf_engine, tokenizer, context):
        super(ACNoneShareDeepSpeedSEPModule, self).__init__(rlhf_engine, tokenizer, context)
        self._is_sep_model = False
        if hasattr(rlhf_engine, 'pred_actor'):
            self._is_sep_model = True
            self.pred_actor_model = self.rlhf_engine.pred_actor
            self.pred_critic_model = self.rlhf_engine.pred_critic
        self.critic_model = self.rlhf_engine.critic
        self._critic_data_loader = None
        self._actor_data_loader = None
        self._last_e2e_time = None

    def generate_experience(self, prompts):
        self._generate_experience_by_seq(prompts)
        return {}


    def _generate_experience_by_seq(self, prompts):
        forward_use_cache = self.context.model_conf.forward_use_cache
        # seq = seq_dict['all_tokens']
        with torch.no_grad():
            gen_start_time = time.time()
            output = self.pred_actor_model.forward_step(prompts)
            gen_exp_time = time.time() - gen_start_time
            if self.pred_actor_model.replicas[0].module is not None:
                print_throughput_step3_sep(gen_exp_time=gen_exp_time)
            output_ref = self.ref_model.forward_step()
            pred_critic = self.critic_model
            if self._is_sep_model:
                pred_critic = self.pred_critic_model

            values = pred_critic.forward_step()

            reward_score = self.reward_model.forward_step()

        return output, output_ref, reward_score, values

    def train_onestep(self, train_epoch):
        """也可以通过外层set
        """
        last_time = time.time()
        self.actor_model.train_step(train_epoch, self.actor_data_loader)
        actor_train_time = time.time() - last_time
        if self.actor_model.module is not None:
            print_throughput_step3_sep(actor_train_time=actor_train_time)
        
        last_time = time.time()
        self.critic_model.train_step(train_epoch, self.critic_data_loader)
        critic_train_time = time.time() - last_time
        if self.critic_model.module is not None:
            e2e_time = None
            cur_time = time.time()                    
            if self._last_e2e_time is not None:
                e2e_time = cur_time - self._last_e2e_time            
            print_throughput_step3_sep(critic_train_time=critic_train_time, e2e_time=e2e_time)
            self._last_e2e_time = cur_time

        critic_loss, actor_loss = 0., 0.

        return actor_loss, critic_loss

    def _sync_model(self, train_model, pred_model, sync_name, total_layer):
        global_param_dict = {}

        src_model_ranks = train_model._place_policy.all_pipeline_stage_ranks    # train_model，可能多pipeline stage
        from alignment.rlhf.distributed.distributed_rlhf_sep_engine import DistModel
        if isinstance(pred_model, DistModel):
            # pred_critic, interleave_model_parallel_ranks仅包含first/last stage。预测模型目前不开pipe
            dst_model_ranks = [item._place_policy.interleave_model_parallel_ranks[0] for item in pred_model.replicas]    
        else:
            dst_model_ranks = pred_model._place_policy.interleave_model_parallel_ranks

        # logger.info(f'Get src_model_ranks: {src_model_ranks}, dst_model_ranks: {dst_model_ranks}')

        mappings = []
        

        
        for dst_replicate_id, dst_ranks in enumerate(dst_model_ranks):    # 每个dst的model_parallel_rank组都需要copy
            for pipe_stage, src_ranks in enumerate(src_model_ranks): # 每个src_ranks代表一个stage, 可能包含数据并行
                assert len(src_ranks) % len(dst_ranks) == 0, "" # model并行组，src_rank的TP应是dst_rank TP的整数倍
                divisions = len(src_ranks) // len(dst_ranks) # N个src TP对应一个dst TP
                for cur_dst_id, dst_rank in enumerate(dst_ranks):    
                    for src_inner_id in range(divisions):
                        src_rank = src_ranks[cur_dst_id * divisions + src_inner_id]
                        group_ranks = tuple(sorted([src_rank, dst_rank]))
                        if group_ranks not in MAPPING_RANKS:
                            MAPPING_RANKS[group_ranks] = torch.distributed.new_group(group_ranks)
                        mappings.append((src_rank, dst_rank, pipe_stage, src_inner_id, MAPPING_RANKS[group_ranks]))
        # print(mappings)

        src_pipe_stage = len(src_model_ranks)
        layers_per_state = total_layer // src_pipe_stage

        train_model.sync(mappings, {},
                         global_param_dict,
                         group_name=sync_name,
                         layers_per_state=layers_per_state,
                         src_pipe_stage=src_pipe_stage)

        pred_model.sync({},
                        mappings,
                        global_param_dict,
                        group_name=sync_name,
                        layers_per_state=layers_per_state,
                        src_pipe_stage=src_pipe_stage)

    def sync(self):
        
        self.free_data_loader()
        start_time = time.time()
        actor_num_layers = self.context.runtime_conf.rlhf_sep_config.actor_sep_config.num_layers
        self._sync_model(self.critic_model, self.pred_critic_model, sync_name='critic_sync', total_layer=actor_num_layers)

        critic_num_layers = self.context.runtime_conf.rlhf_sep_config.critic_sep_config.num_layers
        self._sync_model(self.actor_model, self.pred_actor_model, sync_name='actor_sync', total_layer=critic_num_layers)
        if self.actor_model.module is not None:
            logger.info(f'Sync cost time: {time.time() - start_time}')

    @property
    def critic_data_loader(self):
        if self._critic_data_loader is None:
            from alignment.rlhf.data.data_utils import SEPMiniDataset

            train_global_batch_size = self.context.runtime_conf.rlhf_sep_config.train_global_batch_size
            train_micro_batch_size = self.context.runtime_conf.rlhf_sep_config.train_micro_batch_size

            critic_dp = len(self.critic_model._place_policy.all_data_parallel_group_ranks[0])
            assert train_global_batch_size % (train_micro_batch_size * critic_dp) == 0

            self._critic_data_loader = SEPMiniDataset(int(train_global_batch_size // critic_dp), train_micro_batch_size)
        
        return self._critic_data_loader
    
    @property
    def actor_data_loader(self):
        if self._actor_data_loader is None:
            from alignment.rlhf.data.data_utils import SEPMiniDataset

            train_global_batch_size = self.context.runtime_conf.rlhf_sep_config.train_global_batch_size
            train_micro_batch_size = self.context.runtime_conf.rlhf_sep_config.train_micro_batch_size

            actor_dp = len(self.actor_model._place_policy.all_data_parallel_group_ranks[0])
            assert train_global_batch_size % (train_micro_batch_size * actor_dp) == 0

            # 单个DP的batch_size
            self._actor_data_loader = SEPMiniDataset(int(train_global_batch_size // actor_dp), train_micro_batch_size)
        
        return self._actor_data_loader
    
    def set_epoch(self, epoch):
        self.actor_data_loader.set_epoch(epoch)
        self.critic_data_loader.set_epoch(epoch)


    def free_data_loader(self):
        self.actor_data_loader.free()
        self.critic_data_loader.free()


class DeepSpeedModuleUnsupervised(ACNoneShareDeepSpeedModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
