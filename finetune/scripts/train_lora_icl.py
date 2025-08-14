# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
#prevent model from using lab machine cache
os.environ['HF_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/hf_home_cache'
os.environ['XDG_CACHE_HOME'] = '/vol/bitbucket/al4624/cache/general_cache/xdg_cache_home'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
)
import wandb
from peft import LoraConfig, get_peft_model
from core.arguments import parse_args
from core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from core.datasets.gpt_dataset import GPTDatasetConfig, GPTDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_GLOBAL_TOKENIZER = None

def find_tensor_sub_seq(batch_ids, sub_seq_ids):    
    first_token = sub_seq_ids[0]
    batch_size, seq_len = batch_ids.shape
    sub_seq_len = len(sub_seq_ids)
    positions = torch.full((batch_size,), seq_len, dtype=torch.int, device=batch_ids.device)
    for b in range(batch_size):
        seq = batch_ids[b]
        candidate_positions = (seq == first_token).nonzero(as_tuple=True)[0]
        for pos in candidate_positions:
            if pos + sub_seq_len <= seq_len:
                window = seq[pos:pos+sub_seq_len]
                if torch.all(window == sub_seq_ids):
                    positions[b] = pos
                    break
    return positions

class ScheduledSamplingTrainer(Trainer):
    def __init__(self, *args, initial_prob=1.0, final_prob=0.5, total_steps=10000, sor_token_ids=None, eor_token_ids=None, sep_id=69, **kwargs):
        # self.initial_prob = initial_prob
        # self.final_prob = final_prob
        # self.total_steps = total_steps
        # self.current_step = 0
        self.sor_token_ids = sor_token_ids
        self.eor_token_ids = eor_token_ids
        #We also need to teacher force the xcodec marker token, since we know this token can mess up the model 
        self.sep_id = sep_id
        #should just be token 32016
        print(f"[DEBUG] sep id is {self.sep_id}")
        print(f"[DEBUG] sor tokens are {self.sor_token_ids}")
        print(f"[DEBUG] eor tokens are {self.eor_token_ids}")
        super().__init__(*args, **kwargs)

    # def _get_teacher_forcing_prob(self):
    #     """Exponential decay for teacher forcing probability."""
    #     decay_rate = (self.final_prob / self.initial_prob) ** (1/self.total_steps)
    #     return max(self.final_prob, self.initial_prob * (decay_rate ** self.current_step))

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        # self.current_step += 1
        # teacher_prob = self._get_teacher_forcing_prob()
        

        # Prepare inputs
        inputs = self._prepare_inputs(inputs)
        input_ids = inputs["input_ids"]
        # attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels", input_ids.clone())
        
        sor_ids_tensor = torch.tensor(self.sor_token_ids, dtype=torch.int, device=input_ids.device)
        eor_ids_tensor = torch.tensor(self.eor_token_ids, dtype=torch.int, device=input_ids.device)

        #obtain prompt mask to locate prompt tokens
        sor_positions = find_tensor_sub_seq(input_ids, sor_ids_tensor)
        eor_positions = find_tensor_sub_seq(input_ids, eor_ids_tensor)
        prompt_complete_flags = (sor_positions < eor_positions).unsqueeze(1).repeat(1, input_ids.shape[1])
        end_prompt_mask = (torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0) 
                            <= (eor_positions + len(eor_ids_tensor) - 1).unsqueeze(1))
        start_prompt_mask = (torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0) 
                        >= (sor_positions).unsqueeze(1))
        prompt_mask = torch.where(prompt_complete_flags, start_prompt_mask & end_prompt_mask, start_prompt_mask | end_prompt_mask)
        
        #obtain sep_id mask
        sep_mask = input_ids == self.sep_id

        teacher_force_mask = prompt_mask | sep_mask

        # Forward pass with teacher forcing
        outputs = model(**inputs)
        # logits = outputs.logits
        
        # outputs = None
        # # Apply scheduled sampling
        # if teacher_prob < 1.0 and self.state.global_step > 0:
        #     # Sample from model predictions
        #     with torch.no_grad():
        #         outputs = model(**inputs)
        #         logits = outputs.logits
        #         sampled_ids = torch.argmax(logits, dim=-1)
        #     #Never replace prompt tokens or sep token
        #     replace_mask =  (~teacher_force_mask) & (torch.rand_like(input_ids.float()) > teacher_prob).bool()
        #     # Shift right to align predictions with next tokens
        #     shifted_sampled = torch.cat([input_ids[:, :1], sampled_ids[:, :-1]], dim=1)
        #     input_ids = torch.where(replace_mask, shifted_sampled, input_ids)
        #     # Re-run forward pass with mixed inputs
        #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # else:
        #     outputs = model(**inputs)
        
        #Mask loss for teacher force tokens
        labels = labels.masked_fill(teacher_force_mask, -100)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        return loss

def is_dataset_built_on_rank():
    # return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0
    return True

def core_gpt_dataset_config_from_args(args):
    print(f"[DEBUG] train_data_path: {args.train_data_path}")
    print(f"[DEBUG] valid_data_path: {args.valid_data_path}")
    print(f"[DEBUG] test_data_path: {args.test_data_path}")
    print(f"[DEBUG] EOD mask loss: {args.eod_mask_loss}" )
    print(f"[DEBUG] Reset position ids: {args.reset_position_ids}")
    print(f"[DEBUG] Reset attention mask: {args.reset_attention_mask}")
    print(f"[DEBUG] Enable shuffle: {args.enable_shuffle}")

    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=args.retro_return_doc_ids,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        eod_id=_GLOBAL_TOKENIZER.vocab['<EOD>'],
        enable_shuffle=args.enable_shuffle,
    )

def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    global _SOR_IDS
    global _EOR_IDS
    global _SEP_ID
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    model_max_length = 8192
    _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                            args.model_name_or_path, 
                            model_max_length=model_max_length, 
                            padding_side="left")
    print(f"[DEBUG] Vocab size: {_GLOBAL_TOKENIZER.vocab_size}")
    ids = [11221, 263, 4696, 5702, 988, 278, 7256, 10768, 313, 3166, 5993, 29871, 29945, 29900, 29900, 304, 5993, 29871, 29896, 29900, 29900, 29900, 29897, 338, 1034, 14214, 491, 11462, 29892, 5706, 263, 5941, 1873, 310, 278, 5702, 988, 278, 7256, 10768, 7087, 278, 3114, 29892, 11395, 362, 29892, 322, 18178, 29265, 310, 278, 18830, 24611, 313, 11083, 322, 1156, 278, 11462, 511, 4803, 278, 6763, 322, 1095, 24611, 408, 9282, 304, 337, 11433, 278, 4567, 470, 5625, 4063, 4004, 10597, 368, 29892, 5662, 3864, 409, 314, 2222, 9636, 3133, 537, 29889, 13, 29961, 15462, 276, 29962, 8198, 29899, 2481, 847, 17939, 13, 29961, 463, 1076, 29962, 13, 13, 13, 29961, 17662, 29962, 13, 13, 13, 29961, 355, 29962, 13, 13, 518, 2962, 29918, 974, 29918, 5679, 29962, 32001, 32016, 45824, 46025, 45669, 45387, 46053, 46189, 45748, 45387, 46221, 46025, 45872, 45387, 45872, 45669, 46025, 45822, 45748, 45874, 46263, 46095, 45874, 46025, 45362, 46189, 46304, 46025, 45387, 45872, 46304, 45362, 45362, 46269, 45872, 45422, 45406, 46263, 45422, 45797, 45874, 46095, 46111, 45874, 45874, 46304, 45630, 45387, 45387, 46095, 46132, 46189, 46304, 45822, 46269, 45669, 45872, 45782, 46095, 45761, 45748, 45406, 46304, 46025, 46025, 45748, 45362, 46095, 45387, 46111, 45354, 46095, 46304, 46095, 46025, 45782, 45406, 46304, 46304, 45362]
    print(f"[DEBUG]: Sample ids decoded: {_GLOBAL_TOKENIZER.decode(ids)}")
    print(f"[DEBUG] 32016: {_GLOBAL_TOKENIZER.convert_ids_to_tokens([32016])}")
    print(f"[DEBUG] 45798: {_GLOBAL_TOKENIZER.convert_ids_to_tokens([45798])}")

    _SOR_IDS=_GLOBAL_TOKENIZER.encode('[start_of_reference]', add_special_tokens=False)
    _EOR_IDS=_GLOBAL_TOKENIZER.encode('[end_of_reference]', add_special_tokens=False)
    _SEP_ID=_GLOBAL_TOKENIZER.convert_tokens_to_ids(['<xcodec>'])[0]

    return _GLOBAL_TOKENIZER

import random

def build_train_valid_test_datasets(args):
    """Build the train, validation, and test datasets."""
    # torch.set_printoptions(threshold=float('inf'), edgeitems=None, linewidth=200)
    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters = args.eval_iters
    train_val_test_num_samples = [train_samples,
                                 eval_iters * args.global_batch_size,
                                 test_iters * args.global_batch_size]

    logger.info("> Building train, validation, and test datasets...")
    try:
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset,
            train_val_test_num_samples,
            core_gpt_dataset_config_from_args(args)
        ).build()
        logger.info("> Finished creating datasets")
        
        # Debugging: Print a few random examples from the training dataset
        num_examples = min(5, len(train_ds))
        sample_indices = random.sample(range(len(train_ds)), num_examples)
        for i, idx in enumerate(sample_indices):
            example = train_ds[idx]
            print(f"[DEBUG] Random Example {i} (index {idx}):")
            for k, v in example.items():
                if isinstance(v, torch.Tensor):
                    print(f"Tensor [{k}]: shape={v.shape}\n{v}")
                else:
                    print(f"{k}: {v}")

        return train_ds, valid_ds, test_ds
    except Exception as e:
        logger.error(f"Failed to build datasets: {e}")
        raise

def _compile_dependencies():
    """Compile dataset C++ code."""
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        logger.info("> Compiling dataset index builder...")
        try:
            from core.datasets.utils import compile_helpers
            compile_helpers()
            logger.info(
                f">>> Done with dataset index builder. Compilation time: {time.time() - start_time:.3f} seconds"
            )
        except Exception as e:
            logger.error(f"Failed to compile helpers: {e}")
            raise

def setup_distributed_training():
    """Setup distributed training environment."""
    try:
        # Initialize process group for distributed training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        
        if world_size > 1:
            # Multi-GPU setup
            torch.cuda.set_device(local_rank)
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            logger.info(f"Distributed training initialized with world size: {world_size}, local rank: {local_rank}")
        else:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
            # Single GPU setup
            logger.info(f"Running on a single GPU (device {local_rank})")
            torch.cuda.set_device(local_rank)
        
        return local_rank
    except Exception as e:
        logger.error(f"Failed to setup distributed training: {e}")
        raise

def create_and_configure_model(args):
    """Create and configure the model with LoRA."""
    try:
        if args.fp16:
            assert not args.bf16
            args.params_dtype = torch.half
        if args.bf16:
            assert not args.fp16
            args.params_dtype = torch.bfloat16
        logger.info(f"Loading base model from {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=args.params_dtype,
            cache_dir=args.cache_dir
        )
        
        logger.info(f"Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to create and configure model: {e}")
        raise


def main():
    # Setup distributed training
    local_rank = setup_distributed_training()
    
    # Compile dependencies after initializing distributed group
    _compile_dependencies()
    
    # Parse arguments
    args = parse_args()
    
    # Build tokenizer
    _build_tokenizer(args)

    # Build datasets
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(args)
    
    train_steps = args.train_iters * args.global_batch_size // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
    print(f"[DEBUG] Total training steps: {train_steps}")

    # Create and configure model
    model = create_and_configure_model(args)

    vocab_size = model.get_input_embeddings().weight.shape[0]
    print("Embedding vocab size:", vocab_size)
    
    # Setup training arguments
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(args.__dict__, allow_extra_keys=True)[0]
    print(f"[DEBUG] Training arguments: {training_args}")
    
    # Initialize wandb if specified
    is_main_process = torch.distributed.get_rank() == 0
    if args.report_to == "wandb" and is_main_process:
        try:
            wandb.init(
                project=args.wandb_project or "YuE-finetune",
                config=vars(args),
                name=args.run_name
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb.")



    # Create trainer
    trainer = ScheduledSamplingTrainer(
        model=model,
        tokenizer=_GLOBAL_TOKENIZER,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=default_data_collator,
        initial_prob=1.0, 
        final_prob=0.5, 
        total_steps=train_steps,
        sor_token_ids=_SOR_IDS,
        eor_token_ids=_EOR_IDS,
        sep_id=_SEP_ID,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=test_ds)
    logger.info(f"Test results: {test_results}")

    # Save model and tokenizer
    if is_main_process:
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        _GLOBAL_TOKENIZER.save_pretrained(training_args.output_dir)
        logger.info("Training completed successfully")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
