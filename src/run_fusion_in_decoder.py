import glob
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import logging
import glob
import jsonlines
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import recall_score
from typing import Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers, seed_everything
import pdb

from utils_fusion_in_decoder import generate_dataloader

from transformers.generation_utils import BeamHypotheses
# transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    # get_cosine_schedule_with_warmup,
    # get_cosine_with_hard_restarts_schedule_with_warmup,
)

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class T5(pl.LightningModule):

    def __init__(self, cfg: DictConfig, tokenizer: T5Tokenizer):
        super(T5, self).__init__()

        self.hparams = cfg
        self.learning_rate = cfg.optim.learning_rate

        self.config = T5Config.from_pretrained(
            cfg.model.config_name if cfg.model.config_name else cfg.model.model_name,
            cache_dir=cfg.model.cache_dir,
        )

        self.tokenizer = tokenizer

        self.model = T5ForConditionalGeneration.from_pretrained(
            cfg.model.model_name,
            from_tf=False,
            config=self.config,
            cache_dir=cfg.model.cache_dir,
        )

        # # add special tokens
        # additional_special_tokens_dict = {
        #     'additional_special_tokens': ['[[', ']]', '==>']
        # }
        # self.tokenizer.add_special_tokens(additional_special_tokens_dict)
        # self.model.resize_token_embeddings(len(self.tokenizer))

        # calculate total training steps
        # self.t_total = len(self.train_dataloader()) * cfg.trainer.max_epochs
        self.t_total = cfg.trainer.max_steps

        if cfg.optim.warmup_steps < 1.0:
            self.warmup_steps = int(cfg.optim.warmup_steps * self.t_total)
        else:
            self.warmup_steps = int(cfg.optim.warmup_steps)
            assert self.warmup_steps < self.t_total

        logger.info(f'Number of warmup steps: {self.warmup_steps}, total number of training steps: {self.t_total}')


    def forward(self, input_ids, attention_mask, labels):
        # input_ids: (batch_size, num_inputs, max_source_length)
        batch_size = input_ids.size(0)
        input_ids = input_ids.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)
        attention_mask_tmp = attention_mask.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)

        # encode the question + context
        encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask_tmp,
            )

        # concat all the encoder hidden states
        hidden_states = encoder_outputs[0]
        encoder_outputs = (hidden_states.reshape(batch_size, -1, self.config.d_model), *encoder_outputs[1:])
        attention_mask = attention_mask.reshape(batch_size, -1) # (b, num_inputs * max_source_length)

        # fusion-in decoder
        outputs = self.model(input_ids=None,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                labels=labels
            )

        return outputs


    @torch.no_grad()
    def _generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **model_specific_kwargs
    ) -> torch.LongTensor:

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 3, "Input prompt should be of shape (batch_size, num_inputs, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self.model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.model.get_encoder), "{} should be a method".format(self.model.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.model.get_encoder()

            input_ids_tmp = input_ids.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)
            attention_mask_tmp = attention_mask.reshape(-1, self.hparams.data.max_source_length) # (b * num_inputs, max_source_length)

            encoder_outputs: tuple = encoder(input_ids_tmp, attention_mask=attention_mask_tmp)
            encoder_outputs = (encoder_outputs[0].reshape(batch_size, -1, self.config.d_model), *encoder_outputs[1:])

            attention_mask = attention_mask.reshape(batch_size, -1) # (b, num_inputs * max_source_length)
            input_ids = input_ids.reshape(batch_size, -1) # (b, num_inputs * max_source_length)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
            ).type_as(input_ids)
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .type_as(input_ids)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        else:
            output = self.model._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output


    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = (encoder_outputs, None) if encoder_outputs is not None else None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self.model(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self.model._use_cache(outputs, use_cache):
                past = outputs[1]
            if self.model.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.model.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            scores = self.model.postprocess_next_token_scores(
                scores=scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self.model._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            # decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)
            decoded = torch.stack(best).type_as(input_ids)

        return decoded


    def generate(self, input_ids, attention_mask):

        outputs = self._generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.hparams.data.max_generation_length,
            do_sample=False,
            # top_p=0.95,
            num_beams=self.hparams.data.num_beams,
            num_return_sequences=self.hparams.data.num_return_sequences,
        )
        if outputs.size(1) < self.hparams.data.max_generation_length:
            outputs = torch.cat((outputs, torch.zeros(outputs.size(0), self.hparams.data.max_generation_length-outputs.size(1)).type_as(outputs)), dim=1)

        return outputs

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, labels)

        # loss
        loss = outputs[0]

        return {'loss': loss}

    def training_step_end(self, outputs):
        loss = outputs['loss'].mean()
        # logs
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, labels)
        generated = self.generate(input_ids, attention_mask)

        # loss
        loss = outputs[0]

        labels[labels==-100] = 0
        return {'val_loss': loss, 'generations': generated, 'labels': labels}

    def validation_step_end(self, outputs):
        loss  = outputs['val_loss'].mean()
        outputs['val_loss'] = loss

        return outputs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        preds  = torch.cat([x['generations'] for x in outputs], dim=0).cpu().numpy().tolist()
        labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().numpy().tolist()
        for pred, label in zip(preds[:5:self.hparams.num_return_sequences], labels[:5]):
            pred_str = self.tokenizer.decode(pred)
            label_str = self.tokenizer.decode(label)
            logger.info(f"generated: {pred_str}")
            logger.info(f"tgt: {label_str}")

        tensorboard_logs = {'val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, labels = batch

        # fwd
        outputs = self.forward(input_ids, attention_mask, labels)
        generated = self.generate(input_ids, attention_mask)

        # loss
        loss = outputs[0]

        labels[labels==-100] = 0

        assert len(generated) == self.hparams.data.num_return_sequences * len(labels)

        return {'test_loss': loss, 'generations': generated, 'labels': labels}

    def test_step_end(self, outputs):
        loss  = outputs['test_loss'].mean()
        outputs['test_loss'] = loss

        return outputs

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        preds  = torch.cat([x['generations'] for x in outputs], dim=0).cpu().numpy().tolist()
        labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().numpy().tolist()

        assert len(preds) == self.hparams.data.num_return_sequences * len(labels)

        new_labels = []
        for label in labels:
            new_labels.extend([label] * self.hparams.data.num_return_sequences)
        labels = new_labels

        assert len(preds) == len(labels)

        results = []
        for pred, label in zip(preds, labels):
            # pdb.set_trace()
            pred_str = self.tokenizer.decode(pred)
            label_str = self.tokenizer.decode(label)
            results.append(
                {
                    "tgt": label_str,
                    "gen": pred_str,
                }
            )

        if not os.path.exists(self.hparams.data.output_dir):
            os.makedirs(self.hparams.data.output_dir)

        with jsonlines.open(os.path.join(self.hparams.data.output_dir, f"{self.hparams.model.model_name}_{self.hparams.data.max_source_length}_generated_beamsize_{self.hparams.data.num_beams}_size_{self.hparams.data.num_return_sequences}.jsonl"), "w") as writer:
            writer.write_all(results)

        tensorboard_logs = {'test_loss': avg_loss}

        return {'avg_test_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.optim.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.hparams.optim.adam_epsilon)
        scheduler = {
             'scheduler': None,
             'monitor': 'val_loss', # Default: val_loss
             'interval': 'step', # step or epoch
             'frequency': 1
        }

        scheduler['scheduler'] = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total
        )

        return [optimizer], [scheduler]


    # def train_dataloader(self):
    #     train_dataloader = generate_dataloader(
    #         data_dir = self.hparams.data.data_dir,
    #         tokenizer = self.tokenizer,
    #         max_source_length = self.hparams.data.max_source_length,
    #         max_target_length = self.hparams.data.max_target_length,
    #         overwrite_cache = self.hparams.data.overwrite_cache,
    #         mode = "train",
    #         batch_size = self.hparams.optim.train_batch_size,
    #         question_type = self.hparams.data.question_type,
    #         passage_type = self.hparams.data.passage_type,
    #     )
    #     return train_dataloader
    #
    # def val_dataloader(self):
    #     val_dataloader = generate_dataloader(
    #         data_dir = self.hparams.data.data_dir,
    #         tokenizer = self.tokenizer,
    #         max_source_length = self.hparams.data.max_source_length,
    #         max_target_length = self.hparams.data.max_target_length,
    #         overwrite_cache = self.hparams.data.overwrite_cache,
    #         mode = "dev",
    #         batch_size = self.hparams.optim.dev_batch_size,
    #         question_type = self.hparams.data.question_type,
    #         passage_type = self.hparams.data.passage_type,
    #     )
    #     return val_dataloader
    #
    # def test_dataloader(self):
    #     test_dataloader = generate_dataloader(
    #         data_dir = self.hparams.data.data_dir,
    #         tokenizer = self.tokenizer,
    #         max_source_length = self.hparams.data.max_source_length,
    #         max_target_length = self.hparams.data.max_target_length,
    #         overwrite_cache = self.hparams.data.overwrite_cache,
    #         mode = "test",
    #         batch_size = self.hparams.optim.test_batch_size,
    #         question_type = self.hparams.data.question_type,
    #         passage_type = self.hparams.data.passage_type,
    #     )
    #     return test_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'/opt/ml/code/model_arcifacts/fusion_config{args.config}.yaml')

    cfg.data.data_dir = os.environ['SM_CHANNEL_TRAIN']
    cfg.data.output_dir = os.path.join(os.environ['SM_MODEL_DIR'], 'output')
    os.makedirs(cfg.data.output_dir, exist_ok=True)
    cfg.model.checkpoint_dir = os.path.join(os.environ['SM_MODEL_DIR'], 'ckpt')
    os.makedirs(cfg.model.checkpoint_dir, exist_ok=True)

    # set seed
    seed_everything(cfg.optim.seed)

    # checkpoint
    checkpoint_dir = os.path.join(cfg.model.checkpoint_dir, cfg.model.model_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_loss',
        filepath=os.path.join(checkpoint_dir, '{epoch}-{val_loss:.4f}'),
        mode='min',
        save_last=False,
        save_top_k=2,
    )

    tokenizer = T5Tokenizer.from_pretrained(
        cfg.model.tokenizer_name if cfg.model.tokenizer_name else cfg.model.model_name,
        cache_dir=cfg.model.cache_dir,
        use_fast=cfg.model.use_fast,
    )

    model_t5 = T5(cfg, tokenizer)

    if cfg.model.model_checkpoint:
        logger.info(f"Loading the checkpoint {cfg.model.model_checkpoint} and continue training")
        model_checkpoint = torch.load(cfg.model.model_checkpoint, map_location=lambda storage, loc: storage)
        model_dict = model_checkpoint['state_dict']
        model_t5.load_state_dict(model_dict)

    # training and testing
    if cfg.do_train:

        train_dataloader = generate_dataloader(
            data_dir = cfg.data.data_dir,
            tokenizer = tokenizer,
            max_source_length = cfg.data.max_source_length,
            max_target_length = cfg.data.max_target_length,
            overwrite_cache = cfg.data.overwrite_cache,
            mode = "train",
            batch_size = cfg.optim.train_batch_size,
            question_type = cfg.data.question_type,
            passage_type = cfg.data.passage_type,
            enable_sql_supervision = cfg.data.enable_sql_supervision,
            cand_for_each_source = cfg.data.cand_for_each_source,
        )

        dev_dataloader = generate_dataloader(
            data_dir = cfg.data.data_dir,
            tokenizer = tokenizer,
            max_source_length = cfg.data.max_source_length,
            max_target_length = cfg.data.max_target_length,
            overwrite_cache = cfg.data.overwrite_cache,
            mode = "dev",
            batch_size = cfg.optim.dev_batch_size,
            question_type = cfg.data.question_type,
            passage_type = cfg.data.passage_type,
            enable_sql_supervision = cfg.data.enable_sql_supervision,
            cand_for_each_source = cfg.data.cand_for_each_source,
        )

        logger.info("Training starts")
        # tb_logger = loggers.WandbLogger(save_dir=cfg.optim.logging_dir, project='fusion in decoder')
        trainer = pl.Trainer(
            # logger=tb_logger,
            checkpoint_callback=checkpoint_callback,
            **OmegaConf.to_container(cfg.trainer, resolve=True),
        )
        trainer.fit(model_t5, train_dataloader, dev_dataloader)
        # trainer.test(model_t5)

    if cfg.do_eval:

        test_dataloader = generate_dataloader(
            data_dir = cfg.data.data_dir,
            tokenizer = tokenizer,
            max_source_length = cfg.data.max_source_length,
            max_target_length = cfg.data.max_target_length,
            overwrite_cache = cfg.data.overwrite_cache,
            mode = "test",
            batch_size = cfg.optim.test_batch_size,
            question_type = cfg.data.question_type,
            passage_type = cfg.data.passage_type,
            enable_sql_supervision = cfg.data.enable_sql_supervision,
            cand_for_each_source = cfg.data.cand_for_each_source,
        )

        logger.info("Evaluation starts")
        best_checkpoint_file = None
        if cfg.model.model_checkpoint == None:
            # find best checkpoint
            best_val_loss = 10000.
            for checkpoint_file in glob.glob(os.path.join(checkpoint_dir, "*val_loss*.ckpt")):
                try:
                    val_loss = float(checkpoint_file.split('=')[-1].replace(".ckpt", ""))
                except:
                    continue
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_file = checkpoint_file
            logger.info(f"Loading the checkpoint: {best_checkpoint_file}")
        else:
            best_checkpoint_file = cfg.model.model_checkpoint

        # load model
        if best_checkpoint_file is not None:
            best_checkpoint = torch.load(best_checkpoint_file, map_location=lambda storage, loc: storage)
            model_t5.load_state_dict(best_checkpoint['state_dict'])

        # test using Trainer test function
        trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
        trainer.test(model_t5, test_dataloader)


if __name__ == "__main__":
    main()
