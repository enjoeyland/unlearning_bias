def add_arguments(parser):
    # Model arguments
    parser.add_argument("--model_name", default="mt5-base", help="Model name, default mt5-base")
    parser.add_argument("--model", default="google/mt5-base", help="Model to use, default google/mt5-base")
    parser.add_argument("--method", default="original", choices=["original", "sisa", "sisa-retain", "negtaskvector", "finetune"], help="Method to use, default original")
    parser.add_argument("--cache_dir", help="Location of the cache, default None")

    # Data arguments
    parser.add_argument("--task", default="flores", help="Name of the task")
    parser.add_argument("--forget_lang", type=str, nargs="+", default=["en"])
    parser.add_argument("--retain_lang", type=str, nargs="+", default=["en"])
    parser.add_argument("--forget_ratio", type=float, default=0.01, help="Forget ratio, default 0.01")
    parser.add_argument("--forget_num", type=int, default=32, help="Forget number, default 1000")
    parser.add_argument("--retain_multiplier", type=int, default=1, help="Retain multiplier, default 1")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers to use, default 4")
    parser.add_argument("--data_dir", default="data/", help="Location of the data, default data/")

    # Sharding arguments
    parser.add_argument("--shards", default=5, type=int, help="Number of shards to use, default 5")
    parser.add_argument("--slices", default=1, type=int, help="Number of slices to use, default 1")

    # Finetuning arguments
    parser.add_argument("--fit_target", default="both", choices=["both", "forget", "retain"], help="Fit target, default both")

    # Negtaskvector arguments
    parser.add_argument("--forget_scaling_coef", default=1.0, type=float, help="Scaling coefficient, default 1.0")
    parser.add_argument("--retain_scaling_coef", default=0, type=float, help="Scaling coefficient, default 0")

    # Training arguments
    parser.add_argument("--do_train", action="store_true", help="Perform training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_mode", default="online", choices=["disabled", "online", "offline"], help="Wandb mode, default disabled")

    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")

    parser.add_argument("--dp_strategy", default="auto", help="Distributed training strategy, default auto",
                        choices=["auto", "ddp", "fsdp", "deepspeed", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_2_offload", "deepspeed_stage_3", "deepspeed_stage_3_offload"])
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"], help="Optimizer to use, default adamw")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001")
    parser.add_argument("--lr_scheduler_type", default="linear", choices=["linear", "cosine"], help="Learning rate scheduler type, default linear")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Warmup ratio, default 0.1")

    parser.add_argument("--epochs", default=20, type=int, help="Train for the specified number of epochs, default 20")
    parser.add_argument("--world_size", default=1, type=int, help="Number of GPUs to use, default 1")
    parser.add_argument("--per_device_batch_size", default=16, type=int, help="Batch size, default 16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--logging_steps", type=int, default=500)    
    parser.add_argument("--eval_steps", type=float, default=1000, help="Evaluate every n steps, default 1000")
    parser.add_argument("--max_tolerance", type=int, default=3)

    parser.add_argument("--output_dir", type=str, default="checkpoints/")

    parser.add_argument("--do_eval", action="store_true", help="Perform evaluation on the validation set")
    parser.add_argument("--do_test", action="store_true", help="Perform evaluation on the test set")
    
    # Testing arguments
    parser.add_argument("--test_src_lang_only", action="store_true", help="Test source language only")

    args = parser.parse_args()
    assert 2 * args.epochs >= args.slices + 1, "Not enough epochs per slice"
    assert args.load_in_4bit + args.load_in_8bit <= 1, "Only one quantization method can be used"