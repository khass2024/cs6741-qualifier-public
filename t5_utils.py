import os
import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers import T5Tokenizer, T5Model
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    wandb.init(
        project="CS6741-T5-Training",  
        name=args.experiment_name,     
        config={                      
            "optimizer": args.optimizer_type,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler_type": args.scheduler_type,
            "num_warmup_epochs": args.num_warmup_epochs,
            "max_n_epochs": args.max_n_epochs,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
        }
    )


def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    if args.finetune:
        # Load the pre-trained model for fine-tuning
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        print("Loaded pre-trained T5 model for fine-tuning.")
    else:
        # Initialize a T5 model from scratch using the same configuration
        config = T5Config.from_pretrained('t5-small')
        model = T5ForConditionalGeneration(config)
        print("Initialized T5 model from scratch.")

    model = model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pth' if best else 'model_last.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)

    if best:
        print(f"Saved the best model at {checkpoint_path}")
    else:
        print(f"Saved the latest model at {checkpoint_path}")
    return

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pth' if best else 'model_last.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Re-initialize model
    model = initialize_model(args)

    # Load the saved state_dict
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded model from checkpoint: {checkpoint_path}")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

