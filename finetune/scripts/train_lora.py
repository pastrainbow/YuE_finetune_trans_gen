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


# print(f"[DEBUG] transformers source files: {transformers.__file__}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_GLOBAL_TOKENIZER = None

class ScheduledSamplingTrainer(Trainer):
    def __init__(self, *args, initial_prob=1.0, final_prob=0.5, total_steps=10000, **kwargs):
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.total_steps = total_steps
        self.current_step = 0
        super().__init__(*args, **kwargs)

    def _get_teacher_forcing_prob(self):
        """Exponential decay for teacher forcing probability."""
        decay_rate = (self.final_prob / self.initial_prob) ** (1/self.total_steps)
        return max(self.final_prob, self.initial_prob * (decay_rate ** self.current_step))

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        #Update schedule sampling params
        self.current_step += 1
        teacher_prob = self._get_teacher_forcing_prob()

        #get inputs
        inputs = self._prepare_inputs(inputs)

        #[TESTING] move inputs to GPU early to free system RAM and avoid OOM kill
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda", non_blocking=True)

        input_ids = inputs["input_ids"]  # shape: [B, T], dtype: long/int

        # Reset peak memory stats so we can see per-step peaks
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---------- Scheduled sampling (no-grad, inference_mode = leanest) ----------
        if teacher_prob < 1.0 and self.state.global_step > 0:
            # 1) get next-token predictions WITHOUT creating autograd history *and* with minimal overhead
            with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                ss_out = model(**inputs)
                # ss_out.logits: [B, T, V]
                sampled_ids = ss_out.logits.argmax(dim=-1).to(torch.int32)  # [B, T]
            # Immediately free the large logits tensor
            del ss_out

            # 2) In-place mixing to avoid new big tensors (no torch.where on full matrix)
            #    mask on CUDA (no CPU roundtrip), then selectively overwrite positions.
            #    This keeps only *one* clone of input_ids.
            mask = (torch.rand_like(input_ids, dtype=torch.float32) > teacher_prob).bool()  # [B, T] on device
            if mask.any():
                # Shift sampled ids (teacher forcing only applies from pos 1)
                # Create a small, temporary view; avoid building a whole "shifted" copy if mask is sparse
                # Make a working copy of input_ids only if we actually modify it
                mixed = input_ids.clone()  # one clone for the whole step

                # Target tokens that will replace positions 1..T-1
                tgt = sampled_ids[:, :-1].to(device=input_ids.device, dtype=input_ids.dtype)

                m = mask[:, 1:]  # only positions that can be replaced
                if m.any():
                    mixed[:, 1:][m] = tgt[m]

                inputs["input_ids"] = mixed
                # free temps
                del mixed, tgt
            # free temps
            del sampled_ids, mask

        # ---------- Main forward (with grad) ----------
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(**inputs)
        loss = out.loss

        # Cleanup *now* so trainer logging/accumulation won't hold onto tensors
        del out
        del inputs
        torch.cuda.empty_cache()

        # Optional: print mem stats every N steps
        if torch.cuda.is_available() and (self.current_step % 10 == 0 or self.current_step < 5):
            alloc_mb = torch.cuda.memory_allocated() / 1024**2
            reserv_mb = torch.cuda.memory_reserved() / 1024**2
            peak_mb  = torch.cuda.max_memory_allocated() / 1024**2
            stats = torch.cuda.memory_stats()
            active   = stats.get("active_bytes.all.current", 0) / 1024**2
            inactive = stats.get("inactive_split_bytes.all.current", 0) / 1024**2  # fragmentation proxy
            print(
                f"[Step {self.current_step}] "
                f"alloc={alloc_mb:.1f}MB | reserved={reserv_mb:.1f}MB | peak={peak_mb:.1f}MB | "
                f"active={active:.1f}MB | inactive_split={inactive:.1f}MB"
            )
        return loss.detach()


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
        print(f"[DEBUG] Model gradient checkpointing enabled: {model.is_gradient_checkpointing}")
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
    )

    # # Create trainer
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=_GLOBAL_TOKENIZER,
    #     args=training_args,
    #     train_dataset=train_ds,
    #     eval_dataset=valid_ds,
    #     data_collator=default_data_collator,
    # )
    
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
