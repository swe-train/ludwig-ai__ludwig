import logging
import re
from typing import Any, Dict, List, Union

import torch
from torch import nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import CATEGORY, LOGITS, PREDICTIONS, PROBABILITIES, TEXT
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.decoders.utils import extract_generated_tokens
from ludwig.modules.llama_modules import ResBlock
from ludwig.schema.decoders.llm_decoders import (
    CategoryExtractorDecoderConfig,
    MedusaDecoderConfig,
    TextExtractorDecoderConfig
)
from ludwig.utils.medusa_utils import (
    evaluate_posterior,
    generate_candidates,
    generate_medusa_buffers,
    initialize_medusa,
    initialize_past_key_values,
    reset_medusa_mode,
    tree_decoding,
    update_inference_inputs
)
from ludwig.utils.strings_utils import get_tokenizer

logger = logging.getLogger(__name__)


# TODO(Arnav): Refactor to split into strategies like splitters
class Matcher:
    def __init__(self, match: Dict[str, Dict[str, Any]]):
        self.match = match

    def contains(self, decoded_input: str, value: str) -> bool:
        return value in decoded_input

    def regex(self, decoded_input: str, regex_pattern: str) -> bool:
        """Perform a regex match on a given text using a specified regex pattern.

        Parameters:
        text (str): The text to perform the match on.
        regex_pattern (str): The regex pattern to use for the match.

        Returns:
        A list of match objects.
        """
        # Compile the regex pattern
        matches = []
        try:
            regex = re.compile(regex_pattern)
            # Perform the match
            matches = regex.findall(decoded_input)
        except Exception:
            logger.warning(f"Regex pattern {regex_pattern} could not be compiled.")
        # If there is a match, matches is a non-empty list, so we can use this
        # to infer if there was a match or not and return a bool
        return len(matches) > 0

    def __call__(self, decoded_input: str) -> Union[str, None]:
        # Greedy match on first label that matches the input
        for label, label_def in self.match.items():
            label_def_type = label_def["type"]
            label_def_value = label_def["value"]

            if label_def_type == "contains":
                is_match = self.contains(decoded_input, label_def_value)
            elif label_def_type == "regex":
                is_match = self.regex(decoded_input, label_def_value)
            else:
                raise ValueError(
                    f"{label_def_type} is not a valid match `type`. Ludwig "
                    "currently supports `contains` and `regex` match types."
                )

            if is_match:
                return label
        return None


@DeveloperAPI
@register_decoder("text_extractor", [TEXT])
class TextExtractorDecoder(Decoder):
    def __init__(
        self,
        input_size: int,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config
        self.input_size = input_size

        # Tokenizer
        self.tokenizer_type = self.config.tokenizer
        self.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
        self.vocab_file = self.config.vocab_file

        # Load tokenizer required for decoding the output from the generate
        # function of the text input feature for LLMs.
        self.tokenizer = get_tokenizer(self.tokenizer_type, self.vocab_file, self.pretrained_model_name_or_path)
        if hasattr(self.tokenizer, "tokenizer"):
            # Transformer Tokenizers
            self.tokenizer_vocab_size = self.tokenizer.tokenizer.vocab_size
        else:
            # TorchText Tokenizers
            self.tokenizer_vocab_size = len(self.tokenizer.vocab)

        # Maximum number of new tokens that will be generated
        self.max_sequence_length = self.max_new_tokens = self.config.max_new_tokens

    @staticmethod
    def get_schema_cls():
        return TextExtractorDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        # Extract the sequences tensor from the LLMs forward pass
        generated_outputs = extract_generated_tokens(
            raw_generated_output_sequences=inputs,
            llm_model_input_lengths=kwargs.get("llm_model_input_lengths", []),
            max_new_tokens=self.max_new_tokens,
            pad_sequence=True,
        )
        outputs_device = generated_outputs.device

        return {
            PREDICTIONS: generated_outputs,
            # TODO(Arnav): Add support for probabilities and logits
            PROBABILITIES: torch.zeros((len(generated_outputs), self.max_new_tokens, self.tokenizer_vocab_size)).to(
                outputs_device
            ),
            LOGITS: torch.zeros((len(generated_outputs), self.max_new_tokens, self.tokenizer_vocab_size)).to(
                outputs_device
            ),
        }


@DeveloperAPI
@register_decoder("category_extractor", [CATEGORY])
class CategoryExtractorDecoder(Decoder):
    def __init__(
        self,
        decoder_config=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config

        self.input_size = self.config.input_size
        self.fallback_label = self.config.fallback_label
        self.str2idx = self.config.str2idx
        self.vocab_size = len(self.config.str2idx)

        # Create Matcher object to perform matching on the decoded output
        self.matcher = Matcher(self.config.match)

        # Tokenizer
        self.tokenizer_type = self.config.tokenizer
        self.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
        self.vocab_file = self.config.vocab_file

        # Load tokenizer required for decoding the output from the generate
        # function of the text input feature for LLMs.
        self.tokenizer = get_tokenizer(self.tokenizer_type, self.vocab_file, self.pretrained_model_name_or_path)

    @staticmethod
    def get_schema_cls():
        return CategoryExtractorDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        # Extract the sequences tensor from the LLMs forward pass
        generated_outputs = extract_generated_tokens(
            raw_generated_output_sequences=inputs,
            llm_model_input_lengths=kwargs.get("llm_model_input_lengths", []),
            max_new_tokens=None,
            pad_sequence=False,
        )
        outputs_device = generated_outputs.device

        # Decode generated outputs from the LLM's generate function.
        decoded_outputs = self.tokenizer.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

        # Parse labels based on matching criteria and return probability vectors
        matched_labels = []
        probabilities = []
        logits = []
        for output in decoded_outputs:
            output = output.lower()  # Convert to lowercase for matching

            matched_label = self.matcher(output)
            idx = self.str2idx[matched_label] if matched_label in self.str2idx else self.str2idx[self.fallback_label]

            # Append the index of the matched label
            matched_labels.append(idx)

            # Append the probability vector for the matched label
            probability_vec = [0] * self.vocab_size
            probability_vec[idx] = 1
            probabilities.append(probability_vec)

            # TODO(Arnav): Figure out how to compute logits. For now, we return
            # a tensor of zeros.
            logits.append([0] * self.vocab_size)

        return {
            PREDICTIONS: torch.tensor(matched_labels, device=outputs_device),
            PROBABILITIES: torch.tensor(probabilities, dtype=torch.float32, device=outputs_device),
            LOGITS: torch.tensor(logits, dtype=torch.float32, device=outputs_device),
        }
    

@DeveloperAPI
@register_decoder("medusa", [TEXT])
class MedusaDecoder(Decoder):
    """
    Implementation from: https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py
    """
    
    def __init__(
        self,
        input_size: int,
        decoder_config=None,
        base_model=None,
        **kwargs,
    ):
        super().__init__()
        self.config = decoder_config
        self.input_size = input_size

        self.base_model = base_model
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]

        # Tokenizer
        self.tokenizer_type = self.config.tokenizer
        self.pretrained_model_name_or_path = self.config.pretrained_model_name_or_path
        self.vocab_file = self.config.vocab_file

        # Load tokenizer required for decoding the output from the generate
        # function of the text input feature for LLMs.
        self.tokenizer = get_tokenizer(self.tokenizer_type, self.vocab_file, self.pretrained_model_name_or_path)
        if hasattr(self.tokenizer, "tokenizer"):
            # Transformer Tokenizers
            self.tokenizer_vocab_size = self.tokenizer.tokenizer.vocab_size
        else:
            # TorchText Tokenizers
            self.tokenizer_vocab_size = len(self.tokenizer.vocab)

        # Maximum number of new tokens that will be generated
        self.max_sequence_length = self.max_new_tokens = self.config.max_new_tokens

        # Create a list of Medusa heads
        self.num_medusa_heads = self.config.num_medusa_heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * self.config.num_medusa_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(self.config.num_medusa_heads)
            ]
        )

        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_head.to(base_model.dtype).to(base_model.device)

        for i in range(self.num_medusa_heads):
            # Initialize the weights of each medusa_head using the base model's weights
            self.medusa_head[i][-1].weight.data[:] = base_model.lm_head.weight.data[:]

    @staticmethod
    def get_schema_cls():
        return MedusaDecoderConfig

    @property
    def input_shape(self):
        return self.input_size

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}
    
    def generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next
        # next token.
        medusa_choices=[1, 7, 6],
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Source: https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/medusa_model.py

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be
                sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        medusa_topk = medusa_choices[1:]

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        results = []

        for _ in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_topk,
                medusa_buffers["tree_indices"],
                temperature,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            results.append(input_ids[0, input_len:])
            # yield {
            #     "text": self.tokenizer.decode(
            #         input_ids[0, input_len:],
            #         skip_special_tokens=True,
            #         spaces_between_special_tokens=False,
            #         clean_up_tokenization_spaces=True,
            #     )
            # }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
        
        return torch.as_tensor(results)

    def forward(self, inputs: List[torch.Tensor], **kwargs):
        # Extract the sequences tensor from the LLMs forward pass
        generated_outputs = extract_generated_tokens(
            raw_generated_output_sequences=inputs,
            llm_model_input_lengths=kwargs.get("llm_model_input_lengths", []),
            max_new_tokens=self.max_new_tokens,
            pad_sequence=True,
        )
        outputs_device = generated_outputs.device

        return {
            PREDICTIONS: generated_outputs,
            # TODO(Arnav): Add support for probabilities and logits
            PROBABILITIES: torch.zeros((len(generated_outputs), self.max_new_tokens, self.tokenizer_vocab_size)).to(
                outputs_device
            ),
            LOGITS: torch.zeros((len(generated_outputs), self.max_new_tokens, self.tokenizer_vocab_size)).to(
                outputs_device
            ),
        }
