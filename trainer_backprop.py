import os
import argparse
import logging

from torch import nn
from tqdm import tqdm
import torch
import pandas as pd
from omegaconf import OmegaConf
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
                            AutoConfig,
                            AutoFeatureExtractor,
                            Wav2Vec2Model
)
from datasets import load_from_disk
from evotorch.neuroevolution.net import count_parameters

import wandb
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
from utils.utils import (
   preprocess_metadata, get_label_id, DataColletorTrain, CollateWav2vec2, get_feature_vector_attention_mask, preprocess_data, DatasetEVO, predict, CustomSupervisedNE
)
from torch.cuda.amp import autocast, GradScaler

from utils.model import FullyConnectedEvo
import numpy as np

# Logger
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        '--config_path',
        default='config/default.yaml',
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        '--continue_train',
        default=False,
        action='store_true',
        help='If True, continues training using the checkpoint_path parameter'
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    if os.path.isdir(cfg.train.model_checkpoint):
        last_checkpoint = get_last_checkpoint(cfg.train.model_checkpoint)
        print("> Resuming Train with checkpoint: ", last_checkpoint)
    else:
        last_checkpoint = None


    train_df = pd.read_csv(cfg.metadata.train_path)
    val_df = pd.read_csv(cfg.metadata.dev_path)
    train_dataset = preprocess_metadata(cfg=cfg, df=train_df)
    val_dataset = preprocess_metadata(cfg=cfg, df=val_df)

    label2id, id2label, num_labels = get_label_id(dataset=train_dataset, label_column=cfg.metadata.label_column)

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.train.model_checkpoint)
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=last_checkpoint if last_checkpoint else cfg.train.model_checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    print(label2id)

    model = Wav2Vec2Model(config).to(device)

    model.freeze_feature_encoder()

    train_data_collator = DataColletorTrain(
        batch=train_dataset,
        processor=feature_extractor,
        apply_augmentation=cfg.data.apply_augmentation,
        sampling_rate=cfg.data.target_sampling_rate,
        apply_dbfs_norm=cfg.data.apply_dbfs_norm,
        target_dbfs=cfg.data.target_dbfs,
        label2id=label2id
    )

    val_data_collator = DataColletorTrain(
        batch=val_dataset,
        processor=feature_extractor,
        apply_augmentation=cfg.data.apply_augmentation,
        sampling_rate=cfg.data.target_sampling_rate,
        apply_dbfs_norm=cfg.data.apply_dbfs_norm,
        target_dbfs=cfg.data.target_dbfs,
        label2id=label2id
    )

    train_loader = torch.utils.data.DataLoader(
        train_data_collator,
        batch_size=16,
        shuffle=True,
        drop_last=False,
        num_workers=10,
        collate_fn=CollateWav2vec2(feature_extractor)
    )

    val_loader = torch.utils.data.DataLoader(
        val_data_collator,
        batch_size=16,
        shuffle=True,
        drop_last=False,
        num_workers=10,
        collate_fn=CollateWav2vec2(feature_extractor)
    )

    # preprocess_data(data_loader=val_loader, model=model, out_path="preloaded_data/val")

    train_dataset = load_from_disk("preloaded_data/train")
    val_dataset = load_from_disk("preloaded_data/val")

    # print(val_dataset)

    network = FullyConnectedEvo(input_size=1024, num_classes=3)
    print(f'Network has {count_parameters(network)} parameters')

    train_dataset_final = DatasetEVO(batch=train_dataset)
    val_dataset_final = DatasetEVO(batch=val_dataset)

    train_loader_final = torch.utils.data.DataLoader(
        train_dataset_final,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=10
    )

    val_loader_final = torch.utils.data.DataLoader(
        val_dataset_final,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=10
    )

    run = wandb.init(
        project=os.path.basename("test"),
        reinit=True,
        mode="disabled"
    )

    train_model(train_dataloader=train_loader_final, val_dataloader=val_loader_final, model=network)

def train_model(
    train_dataloader,
    val_dataloader,
    model,
    lr: float = 1e-3,
    warmup_rate: float = 0.2,
    epochs: int = 120,
    weights_output_dir: str = 'weights',
    mixed_precision: bool = False,
    early_stopping: bool = False,
    patience = 50
) -> None:

    model.to(device)
    # Create directory to save weights if it does not exist
    os.makedirs(weights_output_dir, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # Save gradients of the weights
    wandb.watch(
        model,
        criterion,
        log='all',
        log_freq=10
    )

    loss_min = np.Inf

    # Initialize early stopping
    current_patience = 0

    # Initialize scheduler
    num_train_steps = int(len(train_dataloader) * epochs)
    num_warmup_steps = int(warmup_rate * epochs * len(train_dataloader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps
    )

    # Initialize mixed precision
    scaler = GradScaler(enabled=mixed_precision)

    model.train()
    for e in tqdm(range(epochs)):
        running_loss = 0
        train_accuracy = 0
        train_f1_score = 0
        for train_batch_count, train_sample in enumerate(tqdm(train_dataloader)):
            # print(train_sample.shape)
            train_audio, train_label = train_sample[0].to(device), train_sample[1].to(device)
            # print(train_audio)
            # print(train_audio.shape)
            optimizer.zero_grad()

            with autocast(enabled=mixed_precision):
                out = model(train_audio)
                # print(out)
                train_loss = criterion(out, train_label)

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Linear schedule with warmup
            if scheduler:
                scheduler.step()

            running_loss += train_loss

            train_accuracy += model_accuracy(train_label, out)
            train_f1_score += model_f1_score(train_label, out)

            # train_batch_count += 1
            # Saves train loss each 2 steps
            if (train_batch_count % 2) == 0:
                wandb.log({"train_loss": train_loss})

        # Validation step
        else:
            val_loss = 0
            val_accuracy = 0
            val_f1_score = 0
            with torch.no_grad():
                model.eval()
                for val_batch_count, val_sample in enumerate(tqdm(val_dataloader)):
                    val_audio, val_label = val_sample[0].to(device), val_sample[1].to(device)

                    out = model(val_audio)

                    loss = criterion(out, val_label)

                    val_accuracy += model_accuracy(val_label, out)
                    val_f1_score += model_f1_score(val_label, out)

                    val_loss += loss

                    # Saves val loss each 2 steps
                    if (val_batch_count % 2) == 0:
                        wandb.log({"val_loss": loss})

            # Log results on wandb
            wandb.log(
                {
                    "train_acc": (train_accuracy/len(train_dataloader))*100,
                    "val_acc": (val_accuracy/len(val_dataloader))*100,
                    "train_f1": train_f1_score/len(train_dataloader)*100,
                    "val_f1": val_f1_score/len(val_dataloader)*100,
                    "epoch": e
                }
            )

            print(
                'Train Accuracy: {:.3f} | Train F1-Score: {:.3f} | Train Loss: {:.6f} | Val Accuracy: {:.3f} | Val F1-Score: {:.3f} | Val loss: {:.6f}'.format(
                    (train_accuracy/len(train_dataloader))*100,
                    (train_f1_score/len(train_dataloader))*100,
                    running_loss/len(train_dataloader),
                    (val_accuracy/len(val_dataloader))*100,
                    (val_f1_score/len(val_dataloader))*100,
                    val_loss/len(val_dataloader)
                )
            )

            # Prints current learning rate value
            print(f"LR: {optimizer.param_groups[0]['lr']}")

            # Saves the model with the lowest val_loss value
            if val_loss/len(val_dataloader) < loss_min:
                print("Validation Loss Decreasead ({:.6f} --> {:.6f}), saving model...\n".format(loss_min, val_loss/len(val_dataloader)))
                loss_min = val_loss/len(val_dataloader)
                torch.save(
                    {
                        'epoch': epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion
                    },
                    os.path.join(
                        weights_output_dir,
                        f'epochs_{epochs}-loss_{loss_min}-epoch_{e}.pth'
                    )
                )

            if early_stopping:
                current_patience += 1

                if current_patience == patience:
                    print("Early Stopping... ")
                    break

            model.train()

def model_accuracy(label, output):
    """
    Calculates the model accuracy considering its outputs.
    ----
    Params:
        output: Model outputs.
        label: True labels.
    Returns:
        Accuracy as a decimal.
    """
    pb = output
    infered = torch.argmax(pb, dim=-1)
    equals = label == infered

    return torch.mean(equals.type(torch.FloatTensor))


def model_f1_score(label, output):
    """
    Calculates the model f1-score considering its outputs.
    ----
    Params:
        output: Model outputs.
        label: True labels.
    Returns:
        F1 score as a decimal.
    """
    pb = output
    infered = torch.argmax(pb, dim=-1)

    return metrics.f1_score(label.detach().cpu().numpy(), infered.detach().cpu().numpy(), average='macro')

if __name__ == '__main__':
    main()