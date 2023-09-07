"""
pytest -vs test/distributed/test_compute_comm_reordering.py::TestComputeCommReorderingMultiProc::test_sink_waits
"""

"""
TODO: unit tests to add:
1. sink_waits()
    TODO: design good # and variaty of compute ops
2. raise_comms()
    TODO: design good # and variaty of compute ops
3. reorder_compute_for_overlap()
    1) only 1 comm
    2) 2+ comms
    TODO: design good # and variaty of compute ops
4. reorder_compute_and_comm_for_overlap() integration test
"""

import functools
import unittest
from unittest.mock import patch
import torch
from torch._C import FileCheck
import torch.distributed._functional_collectives as _functional_collectives
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.utils import same
from torch._dynamo.testing import CompileCounter
from torch.distributed.distributed_c10d import GroupMember
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_distributed import (
    DynamoDistributedSingleProcTestCase,
    DynamoDistributedMultiProcTestCase,
    _dynamo_dist_per_rank_init,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch._inductor.compile_fx import compile_fx as inductor_compile_fx
from torch._inductor.utils import has_triton, run_and_get_triton_code
import torch._dynamo.logging


# def _gather_along_first_dim(x, group):
#     return _functional_collectives.all_gather_tensor(x, 0, group)


# def _reduce_scatter_along_first_dim(x, group):
#     return _functional_collectives.reduce_scatter_tensor(x, "sum", 0, group)


# def _all_reduce(x, group):
#     return _functional_collectives.all_reduce(x, "sum", group)


@requires_nccl()
class TestComputeCommReorderingMultiProc(DynamoDistributedMultiProcTestCase):
    def get_world_trs(self):
        return {
            "tag": "",
            "ranks": list(range(self.world_size)),
            "group_size": self.world_size,
        }

    @property
    def world_size(self) -> int:
        # hack: no matter whether we have 2 or 3 or 4 gpus, just run on 2
        # works around issue with skipif<2 and workers with unpredictable #s gpu
        return 2

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_lt_x_gpu(2)
    @patch.object(torch._inductor.config, "reorder_for_compute_comm_overlap", False)
    # TODO: somehow inductor bg compile threads are causing hangs at exit with distributed work dtor
    @patch.object(torch._inductor.config, "compile_threads", 1)
    def test_sink_waits(self):
        def example(a, *, ranks):
            ar = _functional_collectives.all_reduce(a, "sum", ranks)
            b = torch.matmul(a, a)
            c = torch.relu(a)
            d = torch.matmul(c, c)
            e = d + ar
            return (e,)

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            ranks = list(range(self.world_size))
            inputs = (torch.ones(4, 4, device="cuda") + self.rank,)

            compiled_fn = torch.compile(example)
            code = run_and_get_triton_code(compiled_fn, *inputs, ranks=ranks)
            print(f"code: {code}")
            # FileCheck() \
            #     .check("buf0_inputs = [arg2_1,arg4_1,arg5_1]") \
            #     .check("buf0 = fun_col_impl._all_to_all_single(input=buf0_inputs[0], output_split_sizes=buf0_inputs[1], input_split_sizes=buf0_inputs[2], tag='', ranks=[0, 1], group_size=2)") \
            #     .check("buf1 = buf0") \
            #     .check("i3 = buf1.size(0)") \
            #     .check("buf1 = _wait_tensor(buf1)") \
            #     .run(code)

            eager_out = example(*inputs, ranks=ranks)
            print(f"eager_out: {eager_out}")
            inductor_out = compiled_fn(*inputs, ranks=ranks)
            print(f"inductor_out: {inductor_out}")
            self.assertTrue(same(eager_out, inductor_out, tol=0.001))
