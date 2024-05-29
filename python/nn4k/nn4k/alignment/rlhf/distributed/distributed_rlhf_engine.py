# coding: utf-8

from collections import deque
import functools
import os
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from deepspeed.utils import groups
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from torch.distributed import distributed_c10d
from torch.distributed.distributed_c10d import get_global_rank

from alignment.util.global_vars import global_context
from alignment.rlhf.distributed.model_placement import ModelPlacement
from alignment.rlhf.trainner.app_ds_rlhf_engine import DeepSpeedACEngine
from alignment.api.utils import dist_util
from alignment.rlhf.hooks.profile_train_hook import TraceEventScope
from alignment.app.util import logger
import deepspeed

GLOBAL_MP_GROUP = None
EMPTY_TENSOR_LENGTH = int(1e6)
ACTOR_WORLD_GROUP = None
CRITIC_WORLD_GROUP = None
REF_WORLD_GROUP = None
REWARD_WORLD_GROUP = None

ACTOR_Z3_PARAM_COUNT = 0
CRITIC_Z3_PARAM_COUNT = 0
REF_Z3_PARAM_COUNT = 0
REWARD_Z3_PARAM_COUNT = 0

# for zero3, 用来标记每次forward_func最后一个bs
IN_ACTOR = False
IS_LAST_FORWARD = True

ACTOR_FWD_MODULE_STACK = list()
CRITIC_FWD_MODULE_STACK = list()
REF_FWD_MODULE_STACK = list()
REWARD_FWD_MODULE_STACK = list()

Z3_MODULE_COUNT = [0]
ACTOR_Z3_MODULE_COUNT = [0]
CRITIC_Z3_MODULE_COUNT = [0]
REF_Z3_MODULE_COUNT = [0]
REWARD_Z3_MODULE_COUNT = [0]

SCATTER_QUEUE = deque()

_WORLD_GROUP_DICT = {}

current_model = None


def data_group_ranks_world_info(ranks):
    ranks = sorted(ranks)
    ranks_string = ranks.__str__()
    return ranks_string


def patch_z3_module_func(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""

    def wrapped_fn(self, *args, **kwargs):
        from deepspeed.runtime.zero.parameter_offload import FWD_MODULE_STACK
        origin_fwd_module_stack = FWD_MODULE_STACK
        FWD_MODULE_STACK = self._fwd_module_stack
        origin_moduld_count = Z3_MODULE_COUNT
        Z3_MODULE_COUNT = self._module_count
        ret_val = func(self, *args, **kwargs)
        FWD_MODULE_STACK = origin_fwd_module_stack
        Z3_MODULE_COUNT = origin_moduld_count
        return ret_val

    return wrapped_fn


class CallModelHooks(object):
    # model_hook暂时一个model对应一个
    MODEL_HOOKS = {}

    def __init__(self, call_method, current_model=None):
        self._current_model = current_model
        self._call_method = call_method

    def __enter__(self):
        if self._current_model not in self.MODEL_HOOKS:
            # 看看module传入
            from alignment.rlhf.hooks.model_hook import ModelHook
            self.MODEL_HOOKS[self._current_model] = ModelHook.get_instance(self._current_model)
        self.MODEL_HOOKS[self._current_model].call(f'{self._call_method}_begin')
        return self.MODEL_HOOKS[self._current_model]

    def __exit__(self, exc_type, exc_value, traceback):
        self.MODEL_HOOKS[self._current_model].call(f'{self._call_method}_end')

    def __call__(self, func):
        def wrapped_func(instance, *args, **kwargs):
            # for装饰器，使用instance.current_model patch当前的current_model
            self._current_model = instance._current_model
            with self:
                res = func(instance, *args, **kwargs)
            return res

        return wrapped_func


class SetScopeLastForward(object):
    """for zero3 控制pred last_trace
    该scope下，IN_ACTOR=True时，仅最后一个actor.forward 会reset_step，其他forward_hydra/generate都不需要set
              IN_ACTOR=False时，last forward reset_step
    """

    def __init__(self, scope_last_forward, scope_in_actor=None):
        self._origin_is_last_forward = None
        self._origin_in_actor = None
        self._scope_last_forward = scope_last_forward
        self._scope_in_actor = scope_in_actor

    def __enter__(self):
        global IS_LAST_FORWARD, IN_ACTOR
        self._origin_is_last_forward = IS_LAST_FORWARD
        self._origin_in_actor = IN_ACTOR
        IS_LAST_FORWARD = self._scope_last_forward
        if self._scope_in_actor is not None:
            IN_ACTOR = self._scope_in_actor

    def __exit__(self, exc_type, exc_value, traceback):
        global IS_LAST_FORWARD, IN_ACTOR
        IS_LAST_FORWARD = self._origin_is_last_forward
        IN_ACTOR = self._origin_in_actor

    def __call__(self, func):
        def wrapped_func(instance, *args, **kwargs):
            with self:
                res = func(instance, *args, **kwargs)
            return res

        return wrapped_func


class PatchDistributedEnv(object):
    def __init__(self, current_model=None):
        self._current_model = current_model
        self._origin_get_world_size = dist.get_world_size
        self._origin_get_rank = dist.get_rank
        self._origin_barrier = dist.barrier
        self._origin_get_world_group = dist.get_world_group
        self._origin_broadcast = dist.broadcast
        self._origin_all_reduce = dist.all_reduce

        from deepspeed.runtime.zero.parameter_offload import FWD_MODULE_STACK
        self._origin_fwd_module_stack = FWD_MODULE_STACK
        self._origin_z3_moduld_count = Z3_MODULE_COUNT
        self._origin_clone_world_group = groups._clone_world_group

    def __enter__(self):
        from deepspeed.runtime.zero.parameter_offload import FWD_MODULE_STACK
        from torch.distributed.distributed_c10d import GroupMember

        global Z3_MODULE_COUNT

        if self._current_model == "ref":
            world_group = REF_WORLD_GROUP
            FWD_MODULE_STACK = REF_FWD_MODULE_STACK
            Z3_MODULE_COUNT = REF_Z3_MODULE_COUNT
        elif self._current_model == "reward":
            world_group = REWARD_WORLD_GROUP
            FWD_MODULE_STACK = REWARD_FWD_MODULE_STACK
            Z3_MODULE_COUNT = REWARD_Z3_MODULE_COUNT
        elif self._current_model == "actor":
            world_group = ACTOR_WORLD_GROUP
            FWD_MODULE_STACK = ACTOR_FWD_MODULE_STACK
            Z3_MODULE_COUNT = ACTOR_Z3_MODULE_COUNT
        elif self._current_model == "critic":
            world_group = CRITIC_WORLD_GROUP
            FWD_MODULE_STACK = CRITIC_FWD_MODULE_STACK
            Z3_MODULE_COUNT = CRITIC_Z3_MODULE_COUNT
        else:
            world_group = GroupMember.WORLD

        def patch_world_size(group=None):
            group = world_group if group is None else group
            return self._origin_get_world_size(world_group)

        def patch_get_rank(group=None):
            group = world_group if group is None else group
            return self._origin_get_rank(group)

        def patch_barrier(group=GroupMember.WORLD, async_op=False, device_ids=None):
            group = world_group if group == GroupMember.WORLD else group
            return self._origin_barrier(world_group, async_op, device_ids)

        def patch_get_world_group():
            return world_group

        def patch_broadcast(tensor, src, group=None, async_op=False):
            #/opt/conda/lib/python3.8/site-packages/deepspeed/runtime/zero/utils.py:70 in  get_lst_from_rank0 给的src=0不对
            src_rank = get_global_rank(world_group, 0) # 默认写src0 
            return self._origin_broadcast(tensor, src_rank, group=world_group, async_op=async_op)

        def patch_clone_world_group():
            return world_group

        def patch_all_reduce(*args, **kwargs):
            if kwargs.get('group', None) is None:
                kwargs['group'] = world_group
            return self._origin_all_reduce(*args, **kwargs)

        dist.get_world_size = patch_world_size
        dist.get_rank = patch_get_rank
        dist.barrier = patch_barrier
        dist.get_world_group = patch_get_world_group
        dist.broadcast = patch_broadcast
        dist.all_reduce = patch_all_reduce
        groups._clone_world_group = patch_clone_world_group

    def __exit__(self, exc_type, exc_value, traceback):
        dist.get_rank = self._origin_get_rank
        dist.get_world_size = self._origin_get_world_size
        dist.barrier = self._origin_barrier
        dist.get_world_group = self._origin_get_world_group
        dist.broadcast = self._origin_broadcast
        dist.all_reduce = self._origin_all_reduce
        groups._clone_world_group = self._origin_clone_world_group

        from deepspeed.runtime.zero.parameter_offload import FWD_MODULE_STACK
        FWD_MODULE_STACK = self._origin_fwd_module_stack
        global Z3_MODULE_COUNT
        Z3_MODULE_COUNT = self._origin_z3_moduld_count

    def __call__(self, func):
        def wrapped_func(instance, *args, **kwargs):
            # for装饰器，使用instance.current_model patch当前的current_model
            self._current_model = instance._current_model
            with self:
                res = func(instance, *args, **kwargs)
            return res

        return wrapped_func


class PatchDistData(object):
    def __init__(self):
        self._ori_torch_get_rank = distributed_c10d.get_rank

    def __enter__(self):
        def patch_get_rank(*args, **kwargs):
            return self._ori_torch_get_rank()

        distributed_c10d.get_rank = patch_get_rank

    def __exit__(self, exc_type, exc_value, traceback):
        distributed_c10d.get_rank = self._ori_torch_get_rank

    def __call__(self, func):
        def wrapped_func(instance, *args, **kwargs):
            with self:
                res = func(instance, *args, **kwargs)
            return res

        return wrapped_func


def patch_deepspeed_groups_clone_world_group(patch_broadcast_src_rank=False):
    def patch_get_broadcast_src_rank():
        root_rank = dist.get_global_rank(groups._get_data_parallel_group(), 0)

        print("root_rank is : {}".format(root_rank))
        return root_rank

    def patch_clone_world_group():

        assert dist.is_initialized(), "dist is not initialized"

        global current_model
        global ACTOR_WORLD_GROUP
        global CRITIC_WORLD_GROUP
        global REF_WORLD_GROUP
        global REWARD_WORLD_GROUP

        if current_model == "ref":
            return REF_WORLD_GROUP
        elif current_model == "reward":
            return REWARD_WORLD_GROUP
        elif current_model == "actor":
            return ACTOR_WORLD_GROUP
        elif current_model == "critic":
            return CRITIC_WORLD_GROUP

    groups._clone_world_group = patch_clone_world_group
    if patch_broadcast_src_rank:
        groups._get_broadcast_src_rank = patch_get_broadcast_src_rank

    from deepspeed.runtime.zero.partition_parameters import Init
    z3_origin_init = Init.__init__

    def patch_z3_param_init(*args, **kwargs):

        from alignment.rlhf.distributed.distributed_rlhf_engine import current_model
        global REF_Z3_PARAM_COUNT
        global ACTOR_Z3_PARAM_COUNT
        global CRITIC_Z3_PARAM_COUNT
        global REWARD_Z3_PARAM_COUNT
        if current_model == 'ref':
            Init.param_id = REF_Z3_PARAM_COUNT
        elif current_model == 'actor':
            Init.param_id = ACTOR_Z3_PARAM_COUNT
        elif current_model == 'critic':
            Init.param_id = CRITIC_Z3_PARAM_COUNT
        elif current_model == 'reward':
            Init.param_id = REWARD_Z3_PARAM_COUNT
        else:
            raise ValueError('Not supported yet')
        print(f'Get count {Init.param_id}')
        return z3_origin_init(*args, **kwargs)

    origin_exit = Init.__exit__

    def patch_exit(*args, **kwargs):
        res = origin_exit(*args, **kwargs)
        global REF_Z3_PARAM_COUNT
        global ACTOR_Z3_PARAM_COUNT
        global CRITIC_Z3_PARAM_COUNT
        global REWARD_Z3_PARAM_COUNT
        from alignment.rlhf.distributed.distributed_rlhf_engine import current_model
        if current_model == 'ref':
            REF_Z3_PARAM_COUNT = Init.param_id
        elif current_model == 'actor':
            ACTOR_Z3_PARAM_COUNT = Init.param_id
        elif current_model == 'critic':
            CRITIC_Z3_PARAM_COUNT = Init.param_id
        elif current_model == 'reward':
            REWARD_Z3_PARAM_COUNT = Init.param_id
        else:
            raise ValueError('Not supported yet')
        print(f'Set count {Init.param_id}')
        return res

    Init.__exit__ = patch_exit
    Init.__init__ = patch_z3_param_init

    from deepspeed.utils import instrument_w_nvtx

    def setup_zero_stage3_hooks(self):
        self.hierarchy = 0

        # reset step if in inference mode
        @instrument_w_nvtx
        def _end_of_forward_hook(module, *args):
            # TODO: 没patch的forward也得考虑下。patch改下
            from alignment.rlhf.distributed.distributed_rlhf_engine import IS_LAST_FORWARD, IN_ACTOR
            # logger.info(f'Get rank {torch.distributed.get_rank()}, {IS_LAST_FORWARD} {IN_ACTOR}')
            if not torch._C.is_grad_enabled() and (not IN_ACTOR or IS_LAST_FORWARD):
                self.get_param_coordinator(training=False).reset_step()

        from alignment.rlhf.distributed.distributed_rlhf_engine import Z3_MODULE_COUNT
        # likely one of them should be enough but just to be safe
        self._register_hooks_recursively(self.module, Z3_MODULE_COUNT)
        self.module.register_forward_hook(_end_of_forward_hook)

        # Add top module to stack trace
        from deepspeed.runtime.zero.parameter_offload import FWD_MODULE_STACK
        FWD_MODULE_STACK.append(self.module)

    from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
    DeepSpeedZeRoOffload.setup_zero_stage3_hooks = setup_zero_stage3_hooks


class GroupPlacePolicy:
    def __init__(self, my_rank, dst_model_ranks, dist_groups=None, group_ranks=None):
        from alignment.api.utils.dist_util import get_total_world_size
        self._total_world_size = get_total_world_size()
        if len(dst_model_ranks) == self._total_world_size:
            logger.info(f'Get dst_model_ranks: {dst_model_ranks} equals to world_size')

        # other policy?
        self._dst_model_ranks = sorted(dst_model_ranks)
        self._other_model_ranks = list(sorted(set(range(self._total_world_size)) - set(self._dst_model_ranks)))

        # 加下dst_model_ranks至少1
        self._group_size = int(len(self._other_model_ranks) // len(self._dst_model_ranks))
        if len(self._other_model_ranks) % len(self._dst_model_ranks):
            logger.warning(f'Other ranks not divisible by dst_model_ranks')
            self._group_size += 1

        self._dist_groups = dist_groups

        self._owner_rank_id = None
        self._is_owner = my_rank in self._dst_model_ranks
        self._my_rank = my_rank
        self._group_ranks = group_ranks

    @property
    def owner_rank_id(self):
        if self._owner_rank_id is None:
            if self.is_owner:
                self._owner_rank_id = self._my_rank
            else:
                assert self._my_rank in self._other_model_ranks
                other_rank_id = self._other_model_ranks.index(self._my_rank)
                self._owner_rank_id = self._dst_model_ranks[int(other_rank_id // self._group_size)]

        return self._owner_rank_id

    @property
    def device_id(self):
        return torch.cuda.current_device()

    @property
    def owner_group_id(self):
        return self.group_ranks.index(self.owner_rank_id)

    @property
    def is_owner(self):
        return self._is_owner

    @property
    def dist_groups(self):
        if self._dist_groups is None:
            print(self.group_ranks)
        return self._dist_groups

    @property
    def group_ranks(self):
        if self._group_ranks is None:
            for i in range(len(self._dst_model_ranks)):
                owner_rank = self._dst_model_ranks[i]

                ranks = [owner_rank]
                ranks.extend([
                    self._other_model_ranks[r]
                    for r in range(i * self._group_size, min((i + 1) * self._group_size, len(self._other_model_ranks)))
                ])
                ranks = list(sorted(ranks))

                # 是否可复用？
                dist_groups = torch.distributed.new_group(ranks=ranks)
                if self.owner_rank_id in ranks:
                    self._group_ranks = ranks
                    self._dist_groups = dist_groups

            print(f'group_ranks {self._group_ranks}'
                  f'owner_rank_id: {self.owner_rank_id}, device_id {self.device_id}')

        return self._group_ranks


class DistributedModuleEngine(nn.Module):
    def __init__(self, module, place_policy, current_model):

        # 这里的module，应该是deepspeed的engine。
        super(DistributedModuleEngine, self).__init__()
        self.module = module
        self._place_policy = place_policy
        self._current_model = current_model
        self._is_training = False
        self._next_data = None
        self._stream = None
        self._in_generate = False
        self._first_generate = True
        self._async_op_queue = deque()
        self._forward_shape = None
        self._tmp_empty_tensor = None
        self._int64_empty_tensor = None

    def get_forward_bs(self, *inputs, **kwargs):
        if 'input_ids' in kwargs:
            # TODO: assert "其他非tensor"
            return kwargs['input_ids'].shape[0]
        else:
            return inputs[0].shape[0]

    def get_forward_seq_len(self, *inputs, **kwargs):
        if 'input_ids' in kwargs:
            # TODO: assert "其他非tensor"
            return kwargs['input_ids'].shape[1]
        else:
            return inputs[0].shape[1]

    @property
    def tmp_empty_tensor(self):
        model_dtype = global_context().runtime_conf.dtype
        dtype = torch.bfloat16 if model_dtype == 'bf16' else torch.float16
        if self._tmp_empty_tensor is None:
            self._tmp_empty_tensor = torch.empty(EMPTY_TENSOR_LENGTH, dtype=dtype, device=self._place_policy.device_id)
        return self._tmp_empty_tensor

    @property
    def int64_empty_tensor(self):
        if self._int64_empty_tensor is None:
            self._int64_empty_tensor = torch.empty(EMPTY_TENSOR_LENGTH,
                                                   dtype=torch.int64,
                                                   device=self._place_policy.device_id)
        return self._int64_empty_tensor

    @property
    def current_stream(self):
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    @TraceEventScope("gather_device")
    @PatchDistData()
    def _gather_device_data(self, data, run_async=False):
        if not torch.distributed.is_initialized() or len(self._place_policy.group_ranks) == 1:
            if not run_async:
                return [data]
            else:
                return [data], MagicMock()
        # TODO: None数据的处理
        gathered_data = []
        ori_shape = data.shape
        gather_output = self._gather_device_data_list([data.view(-1)], run_async)

        res_tensor_data = gather_output[0]
        gathered_data = [item.view(ori_shape) for item in res_tensor_data]

        if not run_async:
            return gathered_data
        else:
            return gathered_data, gather_output[2]

    @TraceEventScope("gather_device")
    @PatchDistData()
    def _gather_device_data_list(self, tensor_list, run_async=False):
        part_sential = [0]
        for item in tensor_list:
            part_sential.append(part_sential[-1] + item.numel())

        if not torch.distributed.is_initialized() or len(self._place_policy.group_ranks) == 1:
            if not run_async:
                return tensor_list, part_sential
            else:
                return tensor_list, part_sential, MagicMock()

        res_tensor_data, out = [], None
        if self._place_policy.is_owner:
            if tensor_list[0].dtype in [torch.bfloat16, torch.float16]:
                res_data_tensor = self.tmp_empty_tensor.narrow(0, 0, part_sential[-1] *
                                                               len(self._place_policy.group_ranks)).view(-1)
            else:
                res_data_tensor = self.int64_empty_tensor.narrow(
                    0, 0, part_sential[-1] * len(self._place_policy.group_ranks)).view(-1)
            res_tensor_data = list(torch.chunk(res_data_tensor, len(self._place_policy.group_ranks)))

            beg = self._place_policy.owner_group_id * part_sential[-1]
            out = res_data_tensor.narrow(0, beg, part_sential[-1])
        else:
            out = None

        concat_tensor = torch.cat(tensor_list, out=out)

        run_scope = MagicMock()
        if run_async:
            self.current_stream.wait_stream(torch.cuda.current_stream())
            run_scope = get_accelerator().stream(self.current_stream)
        with run_scope:
            from deepspeed.comm.comm import gather
            work_op = gather(concat_tensor,
                             res_tensor_data,
                             dst=self._place_policy.owner_rank_id,
                             group=self._place_policy.dist_groups,
                             async_op=run_async)
        if not run_async:
            return res_tensor_data, part_sential
        else:
            return res_tensor_data, part_sential, work_op

    @TraceEventScope("all_gather_device")
    @PatchDistData()
    def _all_gather_device_data_list(self, tensor_list, run_async=False, group=None):
        part_sential = [0]
        for item in tensor_list:
            part_sential.append(part_sential[-1] + item.numel())

        if not torch.distributed.is_initialized() or len(self._place_policy.group_ranks) == 1:
            if not run_async:
                return tensor_list, part_sential
            else:
                return tensor_list, part_sential, MagicMock()

        res_data_tensor = self.int64_empty_tensor.narrow(0, 0, part_sential[-1] *
                                                         len(self._place_policy.group_ranks)).view(-1)
        res_tensor_data = list(torch.chunk(res_data_tensor, len(self._place_policy.group_ranks)))

        beg = self._place_policy.owner_group_id * part_sential[-1]
        out = res_data_tensor.narrow(0, beg, part_sential[-1])

        concat_tensor = torch.cat(tensor_list, out=out)

        run_scope = MagicMock()
        if run_async:
            self.current_stream.wait_stream(torch.cuda.current_stream())
            run_scope = get_accelerator().stream(self.current_stream)
        with run_scope:
            from deepspeed.comm.comm import all_gather
            work_op = all_gather(res_tensor_data,
                                 concat_tensor,
                                 group=group or self._place_policy.dist_groups,
                                 async_op=run_async)
        if not run_async:
            return res_tensor_data, part_sential
        else:
            return res_tensor_data, part_sential, work_op

    @TraceEventScope("scatter_device")
    @PatchDistData()
    def _scatter_device_data(self, all_data_list, dtype=torch.float16):
        # dtype类型后续需传递
        if not torch.distributed.is_initialized() or len(self._place_policy.group_ranks) == 1:
            return all_data_list[0]

        data_list = [None]
        # scatter_object_list使用group判断src
        torch.distributed.scatter_object_list(data_list,
                                              all_data_list,
                                              src=self._place_policy.owner_rank_id,
                                              group=self._place_policy.dist_groups)
        # logger.info(f'Get data {data_list}, input is {all_data_list}')
        del all_data_list
        return data_list[0].to(f'cuda:{self._place_policy.device_id}')

    @TraceEventScope("scatter_device")
    @PatchDistData()
    def _scatter_device_data_with_shape(self,
                                        all_data_list,
                                        dtype=torch.float16,
                                        run_async=False,
                                        shape=None,
                                        res_tensor=None):
        # dtype类型后续需传递
        if not torch.distributed.is_initialized() or len(self._place_policy.group_ranks) == 1:
            return all_data_list[0]

        # TODO: 从ds_config透出fp16
        if res_tensor is not None:
            data = res_tensor
        else:
            if self._place_policy.is_owner:
                data = all_data_list[self._place_policy.owner_group_id]
            else:
                data = torch.empty(shape, device=f'cuda:{self._place_policy.device_id}', dtype=dtype)

        run_scope = MagicMock()
        if run_async:
            self.current_stream.wait_stream(torch.cuda.current_stream())
            run_scope = get_accelerator().stream(self.current_stream)
        with run_scope:
            from deepspeed.comm.comm import scatter
            op = scatter(data,
                         all_data_list,
                         group=self._place_policy.dist_groups,
                         src=self._place_policy.owner_rank_id,
                         async_op=run_async)
        if run_async:
            SCATTER_QUEUE.append([op, [all_data_list]])
        return data

    @TraceEventScope("all_to_all_device")
    @PatchDistData()
    def _all_to_all_single(self,
                           input_tensor_list,
                           output_tensor_list,
                           output_split_sizes=None,
                           input_split_sizes=None,
                           run_async=False):
        # dtype类型后续需传递
        if not torch.distributed.is_initialized() or len(self._place_policy.group_ranks) == 1:
            return input_tensor_list[0]

        run_scope = MagicMock()
        if run_async:
            self.current_stream.wait_stream(torch.cuda.current_stream())
            run_scope = get_accelerator().stream(self.current_stream)
        with run_scope:
            from deepspeed.comm.comm import all_to_all_single
            op = all_to_all_single(output_tensor_list,
                                   input_tensor_list,
                                   group=self._place_policy.dist_groups,
                                   output_split_sizes=output_split_sizes,
                                   input_split_sizes=input_split_sizes,
                                   async_op=run_async)
        if run_async:
            SCATTER_QUEUE.append([op, [input_tensor_list]])
        return output_tensor_list

    @CallModelHooks("on_forward")
    def _forward(self, forward_func, last_forward_value, *inputs, **kwargs):
        self._gathered_inputs, self._gathered_kwargs = inputs, kwargs
        kwargs.pop('next_input_ids', None)
        next_input_ids = None
        parse_data_func = kwargs.pop('parse_data_func', None)
        all_gather_data = kwargs.pop('all_gather_data', False)
        disable_gather = kwargs.pop('disable_gather', False)

        res_data_tensor, part_sential = [], [0]
        if not disable_gather and (not self._in_generate or len(self._async_op_queue) == 0):
            # check 整个group确保相同逻辑deque/enque
            with TraceEventScope("first_recv"):

                tensor_data_list = [item.view(-1) for item in inputs if isinstance(item, torch.Tensor)]

                tensor_data_list.extend(
                    [value.contiguous().view(-1) for value in kwargs.values() if isinstance(value, torch.Tensor)])

                if all_gather_data:
                    res_data_tensor, part_sential, async_op = self._all_gather_device_data_list(tensor_data_list,
                                                                                                run_async=True)
                else:
                    res_data_tensor, part_sential, async_op = self._gather_device_data_list(tensor_data_list,
                                                                                            run_async=True)

                self._async_op_queue.append((self._in_generate, async_op))

        res_data = []

        def _wait_queued_data():
            with TraceEventScope("recv_train_data"):
                for i in range(len(self._async_op_queue) - 1, -1, -1):
                    if not self._in_generate and self._async_op_queue[i][0]:
                        continue
                    self._async_op_queue[i][1].wait()
                    del self._async_op_queue[i]

        if self.module is not None:
            for idx in range(len(self._place_policy.group_ranks)):
                if idx == 0:
                    new_inputs, new_kwargs = inputs, kwargs
                else:
                    tensor_idx = idx
                    if idx == self._place_policy.owner_group_id:
                        tensor_idx = 0

                    def _get_tensor(t_idx):
                        res = res_data_tensor[tensor_idx].narrow(0, part_sential[t_idx],
                                                                 part_sential[t_idx + 1] - part_sential[t_idx])
                        # res2 = res_data_tensor[1 - tensor_idx].narrow(
                        #     0, part_sential[t_idx], part_sential[t_idx + 1] - part_sential[t_idx])
                        # if torch.distributed.get_rank() == 5:
                        #     print(
                        #         f'rank: {torch.distributed.get_rank()}, idx: {idx}, tensor_idx: {tensor_idx}, {res} {inputs} {res2}'
                        #     )
                        #     import pdb
                        #     pdb.set_trace()
                        return res

                    new_inputs = [
                        _get_tensor(t_idx).view(item.shape) if isinstance(item, torch.Tensor) else item
                        for t_idx, item in enumerate(inputs)
                    ]
                    num_tensor_in_inputs = len([isinstance(item, torch.Tensor) for item in inputs])
                    new_kwargs = {
                        key: _get_tensor(t_idx + num_tensor_in_inputs).view(value.shape) if isinstance(
                            value, torch.Tensor) else value
                        for t_idx, (key, value) in enumerate(kwargs.items())
                    }
                is_last_forward = last_forward_value and idx == len(self._place_policy.group_ranks) - 1
                with TraceEventScope("inner_forward_run"), SetScopeLastForward(is_last_forward):
                    cur_item = forward_func(*new_inputs, **new_kwargs)
                    cur_item = parse_data_func(cur_item, new_inputs) if parse_data_func is not None else cur_item

                    res_data.append(cur_item)
                del new_inputs, new_kwargs

                if idx == 0:
                    _wait_queued_data()
                    if self._in_generate and self._next_data is not None:
                        assert not res_data_tensor
                        res_data_tensor = [self._next_data]
                        part_sential.append(self._next_data[0].numel())
            res_data[0], res_data[self._place_policy.owner_group_id] = res_data[
                self._place_policy.owner_group_id], res_data[0]
            # del self._gathered_kwargs, self._gathered_inputs
            del res_data_tensor
        else:
            _wait_queued_data()

        if self._in_generate:
            if next_input_ids is not None:
                # self._next_data, async_op = self._gather_device_data(next_input_ids, run_async=True)
                # self._async_op_queue.append((self._in_generate, async_op))
                pass
            else:
                self._next_data = None

        return res_data

    @PatchDistributedEnv()
    def forward(self, *inputs, **kwargs):
        if self.module is not None:
            if self._is_training:
                return self.module(*inputs, **kwargs)
            else:
                # engine默认调用了sync。 for zero3 是否sync?
                return self.module.module(*inputs, **kwargs)
        else:
            return None

    @PatchDistributedEnv()
    def generate(self, *inputs, **kwargs):
        if self.module is not None:
            return self.module.module.generate(*inputs, **kwargs)
        else:
            return None

    def forward_value(self, *inputs, **kwargs):
        if self.module is not None:
            return self.module.forward_value(*inputs, **kwargs)
        else:
            return None

    @PatchDistributedEnv()
    @CallModelHooks("on_backward")
    def backward(self, loss):
        if self.module is not None:
            return self.module.backward(loss)
        else:
            pass

    @CallModelHooks("on_step")
    def step(self):
        if self.module is not None:
            return self.module.step()
        else:
            pass

    def eval(self):
        if self.module is not None:
            self.module.eval()

    def train(self):
        if self.module is not None:
            self.module.train()


class RewardDistributedModuleEngine(DistributedModuleEngine):
    @TraceEventScope("forward_reward")
    @PatchDistributedEnv()
    def forward_value(self, *inputs, **kwargs):
        run_ref_reward_async = global_context().model_conf.run_ref_reward_async

        def _parse_data(item, inp):
            del_keys = [key for key in item.keys() if key not in ['chosen_end_scores']]
            for key in del_keys:
                del item[key]
            return item['chosen_end_scores']

        kwargs['parse_data_func'] = _parse_data
        if run_ref_reward_async:
            assert self.module is not None
            kwargs['all_gather_data'] = True
        all_chosen_values = self._forward(self.module.forward_value if self.module else None, True, *inputs, **kwargs)

        # print(all_values, all_chosen_values)

        # glm中未产出values, 暂时去掉
        # values = self._scatter_device_data(all_values)

        # logger.info(f'Get rank: {torch.distributed.get_rank()} call reward all_chosen_values {all_chosen_values}')

        if run_ref_reward_async:
            return all_chosen_values
        else:
            self._forward_shape = [self.get_forward_bs(*inputs, **kwargs)]
            chosen_values = self._scatter_device_data_with_shape(all_chosen_values,
                                                                 run_async=True,
                                                                 shape=self._forward_shape)
            return {'chosen_end_scores': chosen_values}


class RefDistributedModuleEngine(DistributedModuleEngine):
    @TraceEventScope("forward_ref")
    @PatchDistributedEnv()
    def __call__(self, *inputs, **kwargs):
        run_ref_reward_async = global_context().model_conf.run_ref_reward_async
        from transformers.modeling_outputs import CausalLMOutputWithPast

        def _parse_data(item, inp):
            # llama需要注释
            if hasattr(item, 'value'):
                del item.value
            if hasattr(item, 'last_hidden_states'):
                del item.last_hidden_states
            from alignment.rlhf.module.rlhf_module import gather_log_probs
            res = gather_log_probs(item.logits[:, :-1, :], inp[0][:, 1:])
            del item.logits
            return res

        kwargs['parse_data_func'] = _parse_data
        if run_ref_reward_async:
            assert self.module is not None
            kwargs['all_gather_data'] = True
        all_logits = self._forward(self.module.__call__ if self.module else None, True, *inputs, **kwargs)

        if run_ref_reward_async:
            return all_logits
        else:
            self._forward_shape = [
                self.get_forward_bs(*inputs, **kwargs),
                self.get_forward_seq_len(*inputs, **kwargs) - 1
            ]

            logits = self._scatter_device_data_with_shape(all_logits, run_async=True, shape=self._forward_shape)

            return CausalLMOutputWithPast(logits=logits)


class ActorDistributedModuleEngine(DistributedModuleEngine):
    @SetScopeLastForward(False, True)
    @TraceEventScope("generate")
    @PatchDistributedEnv()
    def generate(self, *inputs, **kwargs):
        self._in_generate = True
        res_data = self._forward(self.module.generate if self.module else None, False, *inputs, **kwargs)
        self._in_generate = False
        value = self._scatter_device_data(res_data, dtype=torch.int64)
        # print("pid : {}, value {}".format(os.getpid(), value))

        return value

    @SetScopeLastForward(True, True)
    @TraceEventScope("forward_actor")
    @PatchDistributedEnv()
    def forward(self, *inputs, **kwargs):
        if self._is_training or len(self._place_policy.group_ranks) == 1:
            # 训练module TrainModel patch了下
            return super().forward(*inputs, **kwargs)
        return_dict = kwargs.get('return_dict', False)

        # if self.module:
        # print('moduls is', type(self.module))

        # last_forward_value = False if hasattr(self.module, "frozen_head") else True
        # TODO: forzen_head支持
        last_forward_value = True
        res_data = self._forward(self.module.forward if self.module else None, last_forward_value, *inputs, **kwargs)

        has_value_head = global_context().model_conf.need_value_head
        if res_data:
            if has_value_head:
                if return_dict:
                    res_data = [torch.cat((item.logits, torch.unsqueeze(item.value, -1)), -1) for item in res_data]
                else:
                    res_data = [torch.cat((item[0], torch.unsqueeze(item[-1], -1)), -1) for item in res_data]                
            else:
                """目前仅nonshare分支可能no_value_head
                """
                if return_dict:
                    res_data = [item.logits for item in res_data]
                else:
                    res_data = [item[0] for item in res_data]

        res_data = self._scatter_device_data(res_data)
        value = None
        if has_value_head:
            logits, value = torch.split(res_data, (res_data.shape[-1] - 1, 1), dim=-1)
            value = value.squeeze(-1)
        else:
            logits = res_data

        # print("pid : {}, logits {}".format(os.getpid(), logits, value))

        if return_dict:
            from alignment.rlhf.model.modeling_ppo import CausalLMOutputWithValue
            return CausalLMOutputWithValue(logits=logits, value=value)
        else:
            return (logits, value)

    @SetScopeLastForward(False, True)
    @TraceEventScope("forward_hydra_actor")
    @PatchDistributedEnv()
    def forward_hydra(self, *inputs, **kwargs):

        if self._is_training or len(self._place_policy.group_ranks) == 1:
            # 训练module TrainModel patch了下
            return super().forward(*inputs, **kwargs)

        # if self.module:
        #     print('moduls is', type(self.module))

        def _parse_data(item, inp):
            # llama需要注释
            del item.value
            if hasattr(item, 'last_hidden_states'):
                del item.last_hidden_states
            from alignment.rlhf.module.rlhf_module import gather_log_probs
            res = gather_log_probs(item.logits[:, :-1, :], inp[0][:, 1:])
            del item.logits
            return res

        kwargs['parse_data_func'] = _parse_data
        last_forward_value = False
        all_logits = self._forward(self.module.forward_hydra if self.module else None, last_forward_value, *inputs,
                                   **kwargs)

        self._forward_shape = [self.get_forward_bs(*inputs, **kwargs), self.get_forward_seq_len(*inputs, **kwargs) - 1]

        logits = self._scatter_device_data_with_shape(all_logits, run_async=True, shape=self._forward_shape)
        from alignment.rlhf.model.modeling_ppo import CausalLMOutputWithValue
        return CausalLMOutputWithValue(logits=logits)

    @TraceEventScope("backward_actor")
    def backward(self, loss):
        super().backward(loss)

    @TraceEventScope("step_actor")
    def step(self):
        super().step()


class CriticDistributedModuleEngine(DistributedModuleEngine):
    @TraceEventScope("forward_critic")
    @PatchDistributedEnv()
    def forward_value(self, *inputs, **kwargs):
        if self._is_training or len(self._place_policy.group_ranks) == 1:
            # 训练module TrainModel patch了下
            return super().forward_value(*inputs, **kwargs)
        return_value_only = kwargs.get('return_value_only', True)
        assert return_value_only, "Currently, return_value_only must be True"

        # if self.module:
        #     print('moduls is', type(self.module))

        res_data = self._forward(self.module.forward_value if self.module else None, True, *inputs, **kwargs)

        value = self._scatter_device_data(res_data)
        # print("pid : {}, logits {}".format(os.getpid(), value))

        return value


def create_data_model_groups(actor_ranks, critic_ranks, ref_ranks, reward_ranks):
    global ACTOR_WORLD_GROUP
    global CRITIC_WORLD_GROUP
    global REF_WORLD_GROUP
    global REWARD_WORLD_GROUP
    ACTOR_WORLD_GROUP = create_data_group(actor_ranks)
    CRITIC_WORLD_GROUP = create_data_group(critic_ranks)
    REF_WORLD_GROUP = create_data_group(ref_ranks)
    REWARD_WORLD_GROUP = create_data_group(reward_ranks)


def create_data_group(ranks):
    if len(ranks) == 0:
        return None
    ranks_str = data_group_ranks_world_info(ranks)
    print(f'ranks_str: {ranks_str}')
    global _WORLD_GROUP_DICT
    if _WORLD_GROUP_DICT.get(ranks_str, None) is None:
        # If not cloned already, clone the world group
        _WORLD_GROUP = dist.new_group(ranks=ranks)
        _WORLD_GROUP_DICT[ranks_str] = _WORLD_GROUP
    return _WORLD_GROUP_DICT[ranks_str]


class DistributedDeepSpeedACEngine:
    def __init__(self, deep_speed_ac_engine: DeepSpeedACEngine, model_placement: ModelPlacement):
        actor_ranks = model_placement.get_actor_ranks()
        ref_ranks = model_placement.get_init_model_ranks()
        critic_ranks = model_placement.get_critic_ranks()
        reward_ranks = model_placement.get_reward_model_ranks()

        rank = dist_util.get_world_rank()
        print("rank is : {}".format(rank))

        create_data_model_groups(actor_ranks, critic_ranks, ref_ranks, reward_ranks)

        patch_deepspeed_groups_clone_world_group()


        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        inference_tp_size = global_context().runtime_conf.actor_conf.hybrid_engine_conf.inference_tp_size
        mp_group_id = global_rank // inference_tp_size
        num_mp_groups = world_size // inference_tp_size
        for mp_group_id in range(num_mp_groups):
            ranks = list(
                range(mp_group_id * inference_tp_size, \
                    (mp_group_id + 1) * inference_tp_size, \
                    1)
            )
            mp_group = dist.new_group(ranks)
            if global_rank in ranks:
                global GLOBAL_MP_GROUP
                # mp_group is used for broader collective
                GLOBAL_MP_GROUP = mp_group

                

        global current_model
        actor_policy = GroupPlacePolicy(rank, actor_ranks)
        actor_policy.group_ranks

        current_model = "actor"
        enable_actor_init_partition = global_context().runtime_conf.actor_conf.enable_init_partition
        enable_critic_init_partition = global_context().runtime_conf.critic_conf.enable_init_partition
        if rank in actor_ranks:
            print("pid : {}, patch actor ranks : {}".format(os.getpid(), actor_ranks))
            with PatchDistributedEnv(current_model):
                with deepspeed.zero.Init(enabled=enable_actor_init_partition):
                    module = deep_speed_ac_engine.init_actor()
                self.actor = ActorDistributedModuleEngine(module, actor_policy, current_model)
        else:
            self.actor = ActorDistributedModuleEngine(None, actor_policy, current_model)

        current_model = "critic"
        if hasattr(deep_speed_ac_engine, 'init_critic'):
            # for share_engine, no critic_model
            critic_policy = GroupPlacePolicy(rank, critic_ranks)
            critic_policy.group_ranks
            if rank in critic_ranks and hasattr(deep_speed_ac_engine, 'init_critic'):
                print("pid : {}, patch critic ranks : {}".format(os.getpid(), critic_ranks))
                with PatchDistributedEnv(current_model):
                    with deepspeed.zero.Init(enabled=enable_critic_init_partition):
                        module = deep_speed_ac_engine.init_critic()
                    self.critic = CriticDistributedModuleEngine(module, critic_policy, current_model)
            else:
                self.critic = CriticDistributedModuleEngine(None, critic_policy, current_model)

        ref_policy = GroupPlacePolicy(rank, ref_ranks)
        ref_policy.group_ranks
        current_model = "ref"
        if rank in ref_ranks:
            print("pid : {}, patch ref ranks : {}".format(os.getpid(), ref_ranks))

            with PatchDistributedEnv(current_model):
                with deepspeed.zero.Init(enabled=enable_actor_init_partition):
                    module = deep_speed_ac_engine.init_ref(self.actor)
                self.ref = RefDistributedModuleEngine(module, ref_policy, current_model)
        else:
            self.ref = RefDistributedModuleEngine(None, ref_policy, current_model)

        current_model = "reward"

        dist_groups, group_ranks = None, None
        if global_context().model_conf.run_ref_reward_async:
            dist_groups, group_ranks = ref_policy.dist_groups, ref_policy.group_ranks
        reward_policy = GroupPlacePolicy(rank, reward_ranks, dist_groups=dist_groups, group_ranks=group_ranks)
        if rank in reward_ranks:
            print("pid : {}, patch reward ranks : {}".format(os.getpid(), reward_ranks))

            with PatchDistributedEnv(current_model):
                with deepspeed.zero.Init(enabled=enable_critic_init_partition):
                    module = deep_speed_ac_engine.init_reward()
                self.reward = RewardDistributedModuleEngine(module, reward_policy, current_model)
        else:
            self.reward = RewardDistributedModuleEngine(None, reward_policy, current_model)
        current_model = None

        # actor_ema not supported yet!
        self.actor_ema = None
        self._deep_speed_ac_engine = deep_speed_ac_engine


class TrainModel(object):
    def __init__(self, train_model, rlhf_engine, one_loss=False):
        self._train_model = train_model
        self._current_stream = None
        self._next_data = None
        self._async_op_queue = deque()
        self._all_model_flattens = isinstance(rlhf_engine, DeepSpeedACEngine)
        self._one_loss = one_loss

    @property
    def current_stream(self):
        if self._current_stream is None:
            self._current_stream = get_accelerator().Stream()
        return self._current_stream

    def _get_dist_concat_loss(self, losses):
        return losses

    def _get_res_loss(self, losses):
        return losses

    def __call__(self, func):
        def wrapped_func(inputs, next_inputs):
            if self._all_model_flattens:

                if self._one_loss:
                    loss = func(inputs, None)
                    return loss
                else:
                    actor_loss, critic_loss = func(inputs, None)
                    return actor_loss, critic_loss

            self._train_model._is_training = True
            all_losses = []

            gathered_data = {}
            if len(self._async_op_queue) == 0:
                with TraceEventScope("first_recv"):
                    for key, value in inputs.items():
                        if_fp32 = False
                        if value.dtype == torch.float32:
                            value = value.to(torch.float16)
                        gathered_data[key], async_op = self._train_model._gather_device_data(value.contiguous(),
                                                                                                run_async=True)
                        self._async_op_queue.append(async_op)

            def _wait_queued_data():
                with TraceEventScope("recv_train_data"):
                    while len(self._async_op_queue) > 0:
                        self._async_op_queue.popleft().wait()

            owner_group_id = self._train_model._place_policy.owner_group_id
            if self._train_model.module is not None:
                for idx in range(len(self._train_model._place_policy.group_ranks)):
                    if idx == 0:
                        new_inputs = inputs
                    else:
                        tensor_idx = idx
                        if idx == owner_group_id:
                            tensor_idx = 0
                        new_inputs = {key: value[tensor_idx] for key, value in gathered_data.items()}

                    # 多次forward/backward累计grad

                    concat_loss = self._get_dist_concat_loss(func(new_inputs, None))
                    del new_inputs

                    all_losses.append(concat_loss)

                    if idx == 0:
                        _wait_queued_data()
                        if self._next_data is not None:
                            gathered_data = self._next_data
                del gathered_data
                all_losses[0], all_losses[owner_group_id] = all_losses[owner_group_id], all_losses[0]
            else:
                _wait_queued_data()

            if next_inputs is not None:
                gathered_data = {}
                for key, value in next_inputs.items():
                    if value.dtype == torch.float32:
                        value = value.to(torch.float16)                    
                    gathered_data[key], async_op = self._train_model._gather_device_data(value.contiguous(),
                                                                                         run_async=True)
                    self._async_op_queue.append(async_op)
                self._next_data = gathered_data
            else:
                self._next_data = None

            # loss理论上非chief可以不scatter？
            forward_shape = [1]
            from alignment.rlhf.module.ac_share_module import ACShareTrainModel
            if isinstance(self, ACShareTrainModel):
                forward_shape = [2]
            all_loss = self._train_model._scatter_device_data_with_shape([item.to(torch.float16) for item in all_losses], shape=forward_shape)
            self._train_model._is_training = False
            return self._get_res_loss(all_loss)

        return wrapped_func