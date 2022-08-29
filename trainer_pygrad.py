import os
import argparse

from torch import nn
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

import matplotlib.pyplot as plt

from utils.utils import (
   preprocess_metadata, get_label_id, DataColletorTrain, CollateWav2vec2, get_feature_vector_attention_mask, preprocess_data, DatasetEVO, predict, CustomSupervisedNE
)

from utils.model import FullyConnectedEvo
import numpy as np

import pygad.torchga
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    global data_inputs, data_outputs, torch_ga, fc, loss_function
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
        batch_size=2,
        shuffle=True,
        drop_last=False,
        num_workers=10,
        collate_fn=CollateWav2vec2(feature_extractor)
    )

    val_loader = torch.utils.data.DataLoader(
        val_data_collator,
        batch_size=2,
        shuffle=True,
        drop_last=False,
        num_workers=10,
        collate_fn=CollateWav2vec2(feature_extractor)
    )

    # preprocess_data(data_loader=val_loader, model=model, out_path="preloaded_data/val")

    train_dataset = load_from_disk("preloaded_data/train")
    val_dataset = load_from_disk("preloaded_data/train")

    # print(val_dataset)

    network = FullyConnectedEvo(input_size=1024, num_classes=3)
    print(f'Network has {count_parameters(network)} parameters')

    train_dataset_final = DatasetEVO(batch=train_dataset)
    val_dataset_final = DatasetEVO(batch=val_dataset)

    val_loader_final = torch.utils.data.DataLoader(
        val_dataset_final,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=10
    )
    data_inputs = torch.tensor(train_dataset["hidden_states"])
    data_outputs = torch.tensor(train_dataset["label_id"])
    loss_function = nn.CrossEntropyLoss()

    fc = FullyConnectedEvo(input_size=1024, num_classes=3)

    torch_ga = pygad.torchga.TorchGA(model=fc, num_solutions=30)

    num_generations = 250
    num_parents_mating = 5
    initial_population = torch_ga.population_weights

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        initial_population=initial_population,
        fitness_func=fitness_func,
        on_generation=callback_generation,
        init_range_low=-0.0001,
        init_range_high=0.01,
        keep_parents=2,
        crossover_type="scattered",
        crossover_probability=0.2
    )

    ga_instance.run()

def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

def fitness_func(solution, sol_idx):
    predictions = pygad.torchga.predict(
        model=fc,
        solution=solution,
        data=data_inputs
    )

    print(predictions)

    solution_fitness = 1.0 / (loss_function(predictions, data_outputs).detach().numpy() + 0.00000001)

    return solution_fitness

if __name__ == '__main__':
    main()