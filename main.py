import os
import argparse

import torch
import numpy as np
import pandas as pd
from torch import nn
import neptune.new as neptune
from omegaconf import OmegaConf
from datasets import load_from_disk
from evotorch.algorithms import SNES
from evotorch.logging import NeptuneLogger, StdOutLogger
from evotorch.neuroevolution.net import count_parameters
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
            AutoConfig,
            AutoFeatureExtractor,
            Wav2Vec2Model
)

from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from utils.utils import (
    DataColletor,
    CollateWav2vec2,
    DatasetEVO,
    CustomSupervisedNE,
    preprocess_data,
    preprocess_metadata,
    get_label_id,
    predict
)

from utils.model import FullyConnectedEvo

load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument(
        '--train',
        default=False,
        action='store_true',
        help='If True, runs ga'
    )
    parser.add_argument(
        '--test',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--pre_processing',
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    if args.train:
        accs_val = []
        recalls_val = []
        precisions_val = []
        f1_scores_val = []

        accs_test = []
        recalls_test = []
        precisions_test = []
        f1_scores_test = []

        # Load preloaded pre-processed audio signals
        train_dataset = load_from_disk(cfg.metadata.preloaded_train)

        # Load Network to check how many parameters it has
        network = FullyConnectedEvo(
            input_size=cfg.fc_model.input_size,
            num_classes=cfg.fc_model.num_classes
        )
        print(f'Network has {count_parameters(network)} parameters')

        # Runs N experiments
        for experiment in range(cfg.train.num_experiments):
            print(f"Experiment: {experiment+1}/{cfg.train.num_experiments}")

            train_dataset_final = DatasetEVO(batch=train_dataset)

            ser_problem = CustomSupervisedNE(
                train_dataset_final, # Using the dataset specified earlier
                FullyConnectedEvo, # Training the MNIST30K module designed earlier
                nn.CrossEntropyLoss(), # Minimizing CrossEntropyLoss
                network_args={
                    "input_size": cfg.fc_model.input_size,
                    "num_classes": cfg.fc_model.num_classes
                },
                minibatch_size=cfg.genetic_algorithm.minibatch_size, # With a minibatch size of 256
                common_minibatch=True,  # Always using the same minibatch across all solutions on an actor
                num_actors=1,  # The total number of CPUs used
                device=device
            )

            searcher = SNES(
                ser_problem,
                stdev_init=cfg.genetic_algorithm.stdev_init,
                popsize=cfg.genetic_algorithm.population_size
            )

            print("Searcher keys: ")
            print([k for k in searcher.iter_status_keys()])

            run = neptune.init(
                project=os.getenv('NEPTUNE_PROJECT'),
                api_token=os.getenv('NEPTUNE_API_TOKEN')
            )

            run["parameters"] = {
                **cfg.metadata,
                **cfg.fc_model,
                **cfg.genetic_algorithm,
                **cfg.data,
                **cfg.train,
                **cfg.test,
                **cfg.logging
            }

            StdOutLogger(searcher, interval=1)

            NeptuneLogger(
                searcher,
                interval = 1,
                run=run
            )

            searcher.run(cfg.genetic_algorithm.generations)

            run.stop()

            print(searcher.status['center'])
            print(searcher.status['pop_best_eval'])

            net = ser_problem.parameterize_net(searcher.status['center']).cpu()
            net.eval()

            val_dataset = load_from_disk(cfg.metadata.preloaded_val)
            test_dataset = load_from_disk(cfg.metadata.preloaded_test)

            val_dataset_final = DatasetEVO(batch=val_dataset)
            test_dataset_final = DatasetEVO(batch=test_dataset)

            val_loader_final = torch.utils.data.DataLoader(
                val_dataset_final,
                batch_size=8,
                shuffle=True,
                drop_last=False,
                num_workers=10
            )

            test_loader_final = torch.utils.data.DataLoader(
                test_dataset_final,
                batch_size=8,
                shuffle=True,
                drop_last=False,
                num_workers=10
            )

            preds_val, labels_ids_val, test_loss_val = predict(test_dataloader=val_loader_final, model=net)
            preds_test, labels_ids_test, test_loss_test = predict(test_dataloader=test_loader_final, model=net)

            new_labels_ids_val = []
            new_labels_ids_test = []

            for labels_id in labels_ids_val:
                new_labels_ids_val.append(int(labels_id.cpu().detach().numpy()))

            for labels_id in labels_ids_test:
                new_labels_ids_test.append(int(labels_id.cpu().detach().numpy()))

            accs_val.append(accuracy_score(new_labels_ids_val, preds_val))
            recalls_val.append(precision_score(new_labels_ids_val, preds_val, average='macro'))
            precisions_val.append(recall_score(new_labels_ids_val, preds_val, average='macro'))
            f1_scores_val.append(f1_score(new_labels_ids_val, preds_val, average='macro'))

            accs_test.append(accuracy_score(new_labels_ids_test, preds_test))
            recalls_test.append(precision_score(new_labels_ids_test, preds_test, average='macro'))
            precisions_test.append(recall_score(new_labels_ids_test, preds_test, average='macro'))
            f1_scores_test.append(f1_score(new_labels_ids_test, preds_test, average='macro'))

        os.makedirs("scores", exist_ok=True)

        df_val = pd.DataFrame(list(zip(accs_val, recalls_val, precisions_val, f1_scores_val)), columns =['Acc', 'Recall', 'Precision', 'F1_Score'])
        df_test = pd.DataFrame(list(zip(accs_test, recalls_test, precisions_test, f1_scores_test)), columns =['Acc', 'Recall', 'Precision', 'F1_Score'])

        df_val.to_csv(
            f"scores/Val-Pop{cfg.genetic_algorithm.population_size}-batch_{cfg.genetic_algorithm.minibatch_size}-gen_{cfg.genetic_algorithm.generations}.csv",
            index=False
        )
        df_test.to_csv(
            f"scores/Test-Pop{cfg.genetic_algorithm.population_size}-batch_{cfg.genetic_algorithm.minibatch_size}-gen_{cfg.genetic_algorithm.generations}.csv",
            index=False
        )


    if args.pre_processing:
        if os.path.isdir(cfg.train.model_checkpoint):
            last_checkpoint = get_last_checkpoint(cfg.train.model_checkpoint)
            print("> Resuming Train with checkpoint: ", last_checkpoint)
        else:
            last_checkpoint = None


        train_df = pd.read_csv(cfg.metadata.train_path)
        val_df = pd.read_csv(cfg.metadata.dev_path)
        test_df = pd.read_csv(cfg.metadata.test_path)

        train_dataset = preprocess_metadata(df=train_df)
        val_dataset = preprocess_metadata(df=val_df)
        test_dataset = preprocess_metadata(df=test_df)

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

        train_data_collator = DataColletor(
            batch=train_dataset,
            processor=feature_extractor,
            sampling_rate=cfg.data.target_sampling_rate,
            apply_dbfs_norm=cfg.data.apply_dbfs_norm,
            target_dbfs=cfg.data.target_dbfs,
            label2id=label2id
        )

        val_data_collator = DataColletor(
            batch=val_dataset,
            processor=feature_extractor,
            sampling_rate=cfg.data.target_sampling_rate,
            apply_dbfs_norm=cfg.data.apply_dbfs_norm,
            target_dbfs=cfg.data.target_dbfs,
            label2id=label2id
        )

        test_data_collator = DataColletor(
            batch=test_dataset,
            processor=feature_extractor,
            sampling_rate=cfg.data.target_sampling_rate,
            apply_dbfs_norm=cfg.data.apply_dbfs_norm,
            target_dbfs=cfg.data.target_dbfs,
            label2id=label2id
        )

        train_loader = torch.utils.data.DataLoader(
            train_data_collator,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            collate_fn=CollateWav2vec2(feature_extractor)
        )

        val_loader = torch.utils.data.DataLoader(
            val_data_collator,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            collate_fn=CollateWav2vec2(feature_extractor)
        )

        test_loader = torch.utils.data.DataLoader(
            test_data_collator,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=10,
            collate_fn=CollateWav2vec2(feature_extractor)
        )

        preprocess_data(data_loader=train_loader, model=model, out_path=cfg.metadata.preloaded_train)
        preprocess_data(data_loader=val_loader, model=model, out_path=cfg.metadata.preloaded_val)
        preprocess_data(data_loader=test_loader, model=model, out_path=cfg.metadata.preloaded_test)

if __name__ == '__main__':
    main()