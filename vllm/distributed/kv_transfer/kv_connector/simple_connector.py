# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class SimpleConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):

        self.config = config.kv_transfer_config
        self.tp_size = config.parallel_config.tensor_parallel_size

        if self.config.kv_connector == "PyNcclConnector":
            from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import (
                PyNcclPipe)
            logger.info(
                "Initializing PyNcclConfig under kv_transfer_config %s",
                self.config)
        elif self.config.kv_connector == "MooncakeConnector":
            # Check if MOONCAKE_CONFIG_PATH is set
            import os
            use_mooncake_distributed_pipe = os.getenv(
                'MOONCAKE_CONFIG_PATH') is not None

            if not use_mooncake_distributed_pipe:
                raise ValueError(
                    "To use MooncakeConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            else:
                from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import (  # noqa: E501
                    MooncakePipe)
                logger.info(
                    "Initializing MooncakeConfig under kv_transfer_config %s",
                    self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[SimpleBuffer] = None
        self.consumer_buffer: Optional[SimpleBuffer] = None

        self.producer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_data_pipe: Union[PyNcclPipe, MooncakePipe]
        self.producer_signal_pipe: Union[PyNcclPipe, MooncakePipe]
        self.consumer_signal_pipe: Union[PyNcclPipe, MooncakePipe]

        # 2 pipes for every rank in the world
        port_offset_base = 2 * self.config.kv_parallel_size
        self.kv_matchs = self.config.kv_matchs

        self.producer_data_pipes = {}
        self.producer_signal_pipes = {}
        self.producer_buffers = {}

        self.consumer_data_pipes = {}
        self.consumer_signal_pipes = {}
        self.consumer_buffers = {}

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        if self.config.is_kv_producer:
            for index,kv_match in enumerate(self.kv_matchs):
                # if not kv_match[0] == self.config.kv_rank:
                #     continue
                key = self.key(kv_match=kv_match)

                if self.config.kv_connector == "PyNcclConnector":
                    self.producer_data_pipe = PyNcclPipe(
                        local_rank=local_rank,
                        config=self.config,
                        port_offset=port_offset_base+index*2,# + kv_match[0]+1,
                        kv_match=kv_match,
                    )
                    self.producer_signal_pipe = PyNcclPipe(
                        local_rank=local_rank,
                        config=self.config,
                        port_offset=port_offset_base+index*2 +1,# (1+ kv_match[0])*100,
                        device="cpu",
                        kv_match=kv_match,
                    )
                elif self.config.kv_connector == "MooncakeConnector":
                    self.producer_data_pipe = MooncakePipe(
                        local_rank=local_rank,
                        config=self.config,
                    )
                    # We only need to initialize MooncakePipe once
                    self.producer_signal_pipe = self.producer_data_pipe
                if not kv_match[0] == self.config.kv_rank:
                    continue
                self.producer_data_pipes[key] = self.producer_data_pipe
                self.producer_signal_pipes[key] = self.producer_signal_pipe
                # self.producer_buffers[key] = self.producer_buffer

            self.producer_buffer = SimpleBuffer(self.producer_signal_pipes,
                                                self.producer_data_pipes,
                                                self.config.kv_buffer_size)
                
        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder
            for index,kv_match in enumerate(self.kv_matchs):
                # if not kv_match[1] == self.config.kv_rank:
                #     continue
                key = self.key(kv_match=kv_match)

                if self.config.kv_connector == "PyNcclConnector":
                    self.consumer_data_pipe = PyNcclPipe(
                        local_rank=local_rank,
                        config=self.config,
                        port_offset=port_offset_base+index*2, #+(kv_match[1]+1),
                        kv_match=kv_match,
                    )
                    self.consumer_signal_pipe = PyNcclPipe(
                        local_rank=local_rank,
                        config=self.config,
                        port_offset=port_offset_base+index*2 + 1,#(kv_match[1]+1)*100,
                        device="cpu",
                        kv_match=kv_match,
                    )
                elif self.config.kv_connector == "MooncakeConnector":
                    self.consumer_data_pipe = MooncakePipe(
                        local_rank=local_rank,
                        config=self.config,
                    )
                    self.consumer_signal_pipe = self.consumer_data_pipe
                if not kv_match[1] == self.config.kv_rank:
                    continue
                self.consumer_data_pipes[key] = self.consumer_data_pipe
                self.consumer_signal_pipes[key] = self.consumer_signal_pipe
                
            self.consumer_buffer = SimpleBuffer(
                self.consumer_signal_pipes,
                self.consumer_data_pipes,
                self.config.kv_buffer_size,
            )

    def select(self, input_tokens: Optional[torch.Tensor],
               roi: Optional[torch.Tensor],kv_match = None) -> List[Optional[torch.Tensor]]:

        assert self.consumer_buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.consumer_buffer.drop_select(input_tokens, roi,kv_match=kv_match)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor,kv_match = None) -> None:

        assert self.producer_buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.producer_buffer.insert(input_tokens, roi, key, value, hidden,kv_match=kv_match)

    def key(self,kv_match):
        return str(kv_match[0]) + '_' + str(kv_match[1])

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
                                             kv_match = None,
    ) -> None:
        # print(kv_match,'++++++'*20)

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping_flat = model_input.attn_metadata.slot_mapping.flatten()
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        start_layer = model_executable.model.start_layer
        end_layer = model_executable.model.end_layer

        model_config = model_executable.model.config
        num_heads = int(model_config.num_key_value_heads / self.tp_size)
        hidden_size = model_config.hidden_size
        num_attention_heads = model_config.num_attention_heads
        head_size = getattr(model_config, "head_dim",
                            int(hidden_size // num_attention_heads))

        # query_lens contains new KV caches that are added to vLLM.
        # so we will send them to decode instance
        # FIXME(Kuntai): This assume that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You have some decode requests while using "
                               "SimpleConnector. Their KVCache won't be sent.")
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]

            keys, values = [], []

            for layer_id in range(start_layer, end_layer):
                kv_cache = kv_caches[layer_id - start_layer]

                key_cache = kv_cache[0].reshape(-1, num_heads, head_size)
                value_cache = kv_cache[1].reshape(-1, num_heads, head_size)

                current_slot_mapping = slot_mapping_flat[start_pos:end_pos]

                keys.append(key_cache[current_slot_mapping].unsqueeze(0))
                values.append(value_cache[current_slot_mapping].unsqueeze(0))

            keys = torch.cat(keys, dim=0)
            values = torch.cat(values, dim=0)

            self.insert(current_tokens,
                        torch.ones_like(current_tokens,
                                        dtype=bool), keys, values,
                        hidden_or_intermediate_states[start_pos:end_pos],
                        kv_match=kv_match)

        logger.debug("[rank%d]: KV send DONE.", torch.distributed.get_rank())

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        kv_match = None,
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:

        # When bypass_model_exec is set to False, it means that at least for one
        # request its corresponding KV cache or hidden state is missing.
        # In this case we need to do prefilling to recompute missing KV cache
        # and hidden states.
        bypass_model_exec = True

        input_tokens_tensor = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        num_prefill_tokens = model_input.attn_metadata.num_prefill_tokens
        slot_mapping = model_input.attn_metadata.slot_mapping.flatten()

        hidden_or_intermediate_states_for_one_req = []

        input_tokens_list = []
        num_computed_tokens_list = []
        start_pos_list = []

        # enumerate different requests
        # FIXME(Kuntai): This impl assumes that all requests are prefill.
        for idx, slen in enumerate(seq_lens):
            start_pos = sum(seq_lens[:idx])
            end_pos = start_pos + slen

            if start_pos >= num_prefill_tokens:
                # This can happen during inflight batching. See:
                # vllm/worker/model_runner.py::_prepare_model_input_tensors:
                # - input_tokens[:num_prefill_tokens] contains prefill tokens.
                # - input_tokens[num_prefill_tokens:] contains decode tokens.
                logger.warning("You should set --enable_chunked_prefill=False "
                               "and --max_num_batched_tokens "
                               "should be equal to --max_seq_len_to_capture")
                bypass_model_exec = False
                assert start_pos == num_prefill_tokens
                break

            current_tokens = input_tokens_tensor[start_pos:end_pos]
            num_tokens = slen

            # collecting data for rebuilding the input
            input_tokens_list.append(current_tokens)
            start_pos_list.append(start_pos)

            ret = self.select(current_tokens,
                              torch.ones_like(current_tokens, dtype=bool),
                              kv_match=kv_match)
            if ret[0] is None:
                # didn't find any match.
                bypass_model_exec = False
                num_computed_tokens_list.append(0)
                continue

            roi: torch.Tensor = ret[1]
            keys: torch.Tensor = ret[2]
            values: torch.Tensor = ret[3]
            hidden: torch.Tensor = ret[4]

            num_computed_tokens = roi.shape[0]
            num_computed_tokens_list.append(num_computed_tokens)

            # check if both KV cache and the hidden states are received
            # If not, need to redo the forwarding to compute missing states
            if not all([(num_computed_tokens == num_tokens), hidden is not None
                        ]):
                bypass_model_exec = False

            # update the end position based on how many tokens are cached.
            end_pos = start_pos + num_computed_tokens

            # put received KV caches into paged memory
            for i in range(model_executable.model.start_layer,
                           model_executable.model.end_layer):

                kv_cache = kv_caches[i - model_executable.model.start_layer]
                layer = model_executable.model.layers[i]

                key_cache, value_cache = kv_cache[0], kv_cache[1]
                ops.reshape_and_cache_flash(
                    keys[i - model_executable.model.start_layer].to(
                        key_cache.device),
                    values[i - model_executable.model.start_layer].to(
                        value_cache.device),
                    key_cache,
                    value_cache,
                    slot_mapping[start_pos:end_pos],
                    layer.self_attn.attn.kv_cache_dtype,
                    layer.self_attn.attn._k_scale,
                    layer.self_attn.attn._v_scale,
                )

            hidden_or_intermediate_states_for_one_req.append(hidden)

        if not bypass_model_exec:
            # Some of the KV cache is not retrieved
            # Here we will fall back to normal model forwarding
            # But optionally you can adjust model_input so that you only do
            # prefilling on those tokens that are missing KV caches.
            logger.warning(
                "[rank%d]: Failed to receive all KVs and hidden "
                "states, redo model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = None

        else:
            logger.debug(
                "[rank%d]: Successfully received all KVs and hidden "
                "states, skip model forwarding.", torch.distributed.get_rank())
            hidden_or_intermediate_states = torch.cat(
                hidden_or_intermediate_states_for_one_req, dim=0)

        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        for producer_data_pipe in self.producer_data_pipes:
            producer_data_pipe.close()
        for consumer_data_pipe in self.consumer_data_pipes:
            consumer_data_pipe.close()

        if self.config.kv_connector == "PyNcclConnector":
            for producer_signal_pipe in self.producer_signal_pipes:
                producer_signal_pipe.close()
            
            for consumer_signal_pipe in self.consumer_signal_pipes:
                consumer_signal_pipe.close()
            
        elif self.config.kv_connector == "MooncakeConnector":
            # MooncakePipe reuses data_pipe for signal_pipe, so we only have to
            # close the data_pipe.
            pass

