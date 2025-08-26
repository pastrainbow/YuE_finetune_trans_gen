# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
import os
#prevent model from using lab machine cache
os.environ['HF_HOME'] = '/vol/bitbucket/al4624/cache/finetune_cache/hf_home_cache'
os.environ['XDG_CACHE_HOME'] = '/vol/bitbucket/al4624/cache/finetune_cache/xdg_cache_home'
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
from torch.cuda.amp import autocast

DEBUG = True

# if DEBUG: print(f"[DEBUG] transformers source files: {transformers.__file__}")

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
    def __init__(self, *args, 
                 initial_prob=1.0, 
                 final_prob=0.5, 
                 total_steps=10000,
                 decay='exponential', 
                 teacher_force=False,
                 sor_token_ids=None,
                 eor_token_ids=None,
                 sep_id=None,
                 **kwargs):
        assert initial_prob >= final_prob
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.total_steps = total_steps
        self.decay = decay
        self.teacher_force = teacher_force
        self.current_step = 0
        if sor_token_ids:
            self.sor_ids_tensor = torch.tensor(sor_token_ids, dtype=torch.int)
        if eor_token_ids:
            self.eor_ids_tensor = torch.tensor(eor_token_ids, dtype=torch.int)
        if sep_id:
            self.sep_id = sep_id
        super().__init__(*args, **kwargs)

    def _get_teacher_forcing_prob(self):
        if self.current_step >= self.total_steps:
            return self.final_prob
        
        if self.decay == 'exponential':
            # if DEBUG: print(f"[DEBUG] Exponential decay")
            #Exponential decay
        
            decay_factor = (self.final_prob / self.initial_prob) ** (self.current_step / self.total_steps)
            if DEBUG: print(f"Teacher force prob: {self.initial_prob * decay_factor}")
            return self.initial_prob * decay_factor
        elif self.decay == 'linear':
            # if DEBUG: print(f"[DEBUG] Linear decay")
            #Linear decay
            decay = (self.initial_prob - self.final_prob) * (self.current_step / self.total_steps)
            return self.initial_prob - decay
        else:
            raise ValueError(f"[ERROR] Decay type {self.decay} does not exist! Can only be either linear or exponential.")

    def _get_teacher_force_mask(self, input_ids):
        sor_ids_tensor = self.sor_ids_tensor.to(input_ids.device)
        eor_ids_tensor = self.eor_ids_tensor.to(input_ids.device)

        #obtain prompt mask to locate prompt tokens
        sor_positions = find_tensor_sub_seq(input_ids, sor_ids_tensor)
        eor_positions = find_tensor_sub_seq(input_ids, eor_ids_tensor)

        prompt_complete_flags = (sor_positions < eor_positions).unsqueeze(1)
        seq_range = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

        end_prompt_mask = (seq_range
                            <= (eor_positions + len(eor_ids_tensor) - 1)
                            .unsqueeze(1)).bool()
        start_prompt_mask = (seq_range
                        >= sor_positions
                        .unsqueeze(1)).bool()

        prompt_mask = torch.where(
            prompt_complete_flags, 
            start_prompt_mask & end_prompt_mask, 
            start_prompt_mask | end_prompt_mask
        )

        #obtain sep_id mask
        sep_mask = input_ids == self.sep_id

        teacher_force_mask = prompt_mask | sep_mask

        return teacher_force_mask

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Step counter
        self.current_step += 1
        teacher_prob = self._get_teacher_forcing_prob()

        inputs = self._prepare_inputs(inputs)
        input_ids = inputs["input_ids"]

        #Get here, since we need the labels for forward pass
        # labels = inputs.get("labels")

            # Reset peak memory stats so we can see per-step peaks
    #     if torch.cuda.is_available():
    #         torch.cuda.reset_peak_memory_stats()

        # --- Scheduled sampling branch ---
        if teacher_prob < 1.0 and self.state.global_step > 0:
            with torch.inference_mode(): #This is better than no_grad
                ss_out = model(**inputs)
                sampled_ids = ss_out.logits.argmax(dim=-1).to(torch.int32)
                del ss_out

            # Random mask of positions to replace with model predictions
            mask = (torch.rand_like(input_ids, dtype=torch.float32) > teacher_prob).bool()

            if mask.any():
                mixed = input_ids.clone()
                #The model output is shifted right by one step, so we need to shift back
                tgt = sampled_ids[:, :-1].to(device=input_ids.device, dtype=input_ids.dtype)
                m = mask[:, 1:]
                if m.any():
                    #Mixed starts with one ground truth token as the initiation
                    mixed[:, 1:][m] = tgt[m] 
                inputs["input_ids"] = mixed
                del mixed, tgt
            del sampled_ids, mask

        # --- Forward pass with grad ---
        outputs = model(**inputs)

        loss = None

        if self.teacher_force:
            # if DEBUG: print(f"[DEBUG] Prompt teacher forcing")
            # teacher_force_mask = self._get_teacher_force_mask(input_ids)
            # labels = labels.masked_fill(teacher_force_mask, -100)
            # loss = torch.nn.functional.cross_entropy(
            #     outputs.logits.view(-1, outputs.logits.size(-1)),
            #     labels.view(-1),
            #     ignore_index=-100
            # )
            raise ValueError('Prompt teacher forcing not supported yet!')
        else:
            # if DEBUG: print(f"[DEBUG] No prompt teacher forcing")
            loss = outputs.loss

        del inputs
        torch.cuda.empty_cache()

         # Optional: print mem stats every N steps
    #     if torch.cuda.is_available() and (self.current_step % 10 == 0 or self.current_step < 5):
    #         alloc_mb = torch.cuda.memory_allocated() / 1024**2
    #         reserv_mb = torch.cuda.memory_reserved() / 1024**2
    #         peak_mb  = torch.cuda.max_memory_allocated() / 1024**2
    #         stats = torch.cuda.memory_stats()
    #         active   = stats.get("active_bytes.all.current", 0) / 1024**2
    #         inactive = stats.get("inactive_split_bytes.all.current", 0) / 1024**2  # fragmentation proxy
    #         print(
    #             f"[Step {self.current_step}] "
    #             f"alloc={alloc_mb:.1f}MB | reserved={reserv_mb:.1f}MB | peak={peak_mb:.1f}MB | "
    #             f"active={active:.1f}MB | inactive_split={inactive:.1f}MB"
    #         )

        return (loss, outputs) if return_outputs else loss


def is_dataset_built_on_rank():
    # return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0
    return True

def core_gpt_dataset_config_from_args(args):
    if DEBUG:
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
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                            args.model_name_or_path, 
                            model_max_length=args.model_max_length, 
                            padding_side="right")
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
        
        if DEBUG:
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
        model.gradient_checkpointing_enable()
        if DEBUG: print(f"[DEBUG] Model gradient checkpointing enabled: {model.is_gradient_checkpointing}")
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
    if DEBUG: print(f"[DEBUG] Total training steps: {train_steps}")
    
    # Create and configure model
    model = create_and_configure_model(args)

    vocab_size = model.get_input_embeddings().weight.shape[0]
    if DEBUG: print("Embedding vocab size:", vocab_size)
    
    # Setup training arguments
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_dict(args.__dict__, allow_extra_keys=True)[0]
    if DEBUG: print(f"[DEBUG] Training arguments: {training_args}")
    
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

    trainer = None
    if args.schedule_sampling:
        if DEBUG: print(f"[DEBUG] Schedule sampling trainer")
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
            decay=args.scheduled_sampling_decay,
            teacher_force=args.prompt_teacher_force,
        )
    else:
        if DEBUG: print(f"[DEBUG] Default trainer")
        trainer = Trainer(
            model=model,
            tokenizer=_GLOBAL_TOKENIZER,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=default_data_collator,
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
