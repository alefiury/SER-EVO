import os
import operator
import functools
import traceback
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from torch import nn

import torch
import tqdm
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
from pydub import AudioSegment
from omegaconf import DictConfig
from datasets import Dataset, load_metric
from transformers import Wav2Vec2Processor
from audiomentations import AddGaussianNoise, PitchShift
from evotorch.neuroevolution import SupervisedNE, NEProblem


metric = load_metric("f1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    """Computes metric on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids, average="macro")


def preprocess_metadata(df: pd.DataFrame):
    """Maps the real path to the audio files"""
    df.reset_index(drop=True, inplace=True)
    df_dataset = Dataset.from_pandas(df)

    return df_dataset


def get_label_id(dataset: Dataset, label_column: str):
    """Gets the labels IDs"""
    label2id, id2label = dict(), dict()

    labels = dataset.unique(label_column)
    labels.sort()

    num_labels = len(id2label)

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    return label2id, id2label, num_labels


def predict(test_dataloader, model):
    # model.to(device)
    loss = nn.CrossEntropyLoss()
    # for name, param in model.named_parameters():
    #     print(name, param)
    model.eval()
    preds = []
    paths = []
    test_loss = 0
    labels_ids = []
    with torch.no_grad():
        for batch, label in tqdm.tqdm(test_dataloader):
            logits = model(batch)
            test_loss += loss(logits, label).item()
            scores = F.softmax(logits, dim=-1)
            pred = torch.argmax(scores, dim=1).cpu().detach().numpy()

            preds.append(list(pred))
            labels_ids.append(list(label))

    preds = functools.reduce(operator.iconcat, preds, [])
    labels_ids = functools.reduce(operator.iconcat, labels_ids, [])

    return preds, labels_ids, test_loss


def get_feature_vector_attention_mask(feature_vector_length: int, attention_mask: torch.LongTensor):
    output_lengths = get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
    batch_size = attention_mask.shape[0]

    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )
    # these two operations makes sure that all values before the output lengths idxs are attended to
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return attention_mask


def get_feat_extract_output_lengths(input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths


class CollateWav2vec2:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: List) -> Tuple[torch.Tensor, torch.Tensor]:
        input_features = []
        label_features = []
        wav_paths = []
        labels = []

        for element in batch:
            input_features.append({"input_values": element["input_values"]})
            label_features.append(element["label_id"])
            wav_paths.append(element["wav_path"])
            labels.append(element["label"])


        batch_padded = self.processor.pad(
            input_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        return batch_padded, label_features, wav_paths, labels


class DataColletor(torch.utils.data.Dataset):
    def __init__(
        self,
        batch,
        processor: Wav2Vec2Processor,
        sampling_rate: int = 16000,
        apply_dbfs_norm: Union[bool, str] = False,
        target_dbfs: int = 0.0,
        label2id: Dict = None
    ):
        self.batch = batch

        self.processor = processor
        self.sampling_rate = sampling_rate

        self.apply_dbfs_norm = apply_dbfs_norm
        self.target_dbfs = target_dbfs

        self.label2id = label2id

        if self.apply_dbfs_norm:
            print("Applying Normalization... ")

    def __len__(self):
        return self.batch.num_rows

    def __getitem__(self, index: int) -> Dict[torch.Tensor, np.ndarray]:
        try:
            # Gain Normalization
            if self.apply_dbfs_norm:
                # Audio is loaded in a byte array
                sound = AudioSegment.from_file(self.batch[index]["wav_file"], format="wav")
                sound = sound.set_channels(1)
                change_in_dBFS = self.target_dbfs - sound.dBFS
                # Apply normalization
                normalized_sound = sound.apply_gain(change_in_dBFS)
                # Convert array of bytes back to array of samples in the range [-1, 1]
                # This enables to work wih the audio without saving on disk
                norm_audio_samples = np.array(normalized_sound.get_array_of_samples()).astype(np.float32, order='C') / 32768.0

                if sound.channels < 2:
                    norm_audio_samples = np.expand_dims(norm_audio_samples, axis=0)

                # Expand one dimension and convert to torch tensor to have the save output shape and type as torchaudio.load
                speech_array = torch.from_numpy(norm_audio_samples)
                sampling_rate = sound.frame_rate

            # Load wav
            else:
                speech_array, sampling_rate = torchaudio.load(self.batch[index]["wav_file"])
            # Transform to Mono
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

            if sampling_rate != self.sampling_rate:
                transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                speech_array = transform(speech_array)
                sampling_rate = self.sampling_rate

            speech_array = speech_array.squeeze().numpy()
            input_tensor = self.processor(speech_array, sampling_rate=sampling_rate).input_values
            input_tensor = np.squeeze(input_tensor)

        except Exception:
            print("Error during load of audio:", self.batch[index]["wav_file"])

        return {
            "wav_path": self.batch[index]["wav_file"],
            "input_values": input_tensor,
            "label_id": torch.tensor(int(self.label2id[self.batch[index]["label"]])),
            "label": self.batch[index]["label"]
        }


class DataColletorMelSpectogram(torch.utils.data.Dataset):
    def __init__(
        self,
        batch,
        processor: Wav2Vec2Processor,
        apply_augmentation: bool = False,
        sampling_rate: int = 16000,
        apply_dbfs_norm: Union[bool, str] = False,
        target_dbfs: int = 0.0,
        label2id: Dict = None
    ):
        self.batch = batch

        self.processor = processor
        self.sampling_rate = sampling_rate

        self.apply_dbfs_norm = apply_dbfs_norm
        self.target_dbfs = target_dbfs

        self.apply_augmentation = apply_augmentation

        self.label2id = label2id

    def __len__(self):
        return self.batch.num_rows

    def __getitem__(self, index: int) -> Dict[torch.Tensor, np.ndarray]:
        try:
            # Gain Normalization
            if self.apply_dbfs_norm:
                # Audio is loaded in a byte array
                sound = AudioSegment.from_file(self.batch[index]["wav_file"], format="wav")
                sound = sound.set_channels(1)
                change_in_dBFS = self.target_dbfs - sound.dBFS
                # Apply normalization
                normalized_sound = sound.apply_gain(change_in_dBFS)
                # Convert array of bytes back to array of samples in the range [-1, 1]
                # This enables to work wih the audio without saving on disk
                norm_audio_samples = np.array(normalized_sound.get_array_of_samples()).astype(np.float32, order='C') / 32768.0

                if sound.channels < 2:
                    norm_audio_samples = np.expand_dims(norm_audio_samples, axis=0)

                # Expand one dimension and convert to torch tensor to have the save output shape and type as torchaudio.load
                speech_array = torch.from_numpy(norm_audio_samples)
                sampling_rate = sound.frame_rate

            # Load wav
            else:
                speech_array, sampling_rate = torchaudio.load(self.batch[index]["wav_file"])
            # Transform to Mono
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)

            if sampling_rate != self.sampling_rate:
                transform = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                speech_array = transform(speech_array)
                sampling_rate = self.sampling_rate

            speech_array = speech_array.squeeze().numpy()
            fbank = torchaudio.compliance.kaldi.fbank(
                speech_array,
                htk_compat=True,
                sample_frequency=sampling_rate,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=self.melbins,
                dither=0.0,
                frame_shift=10
            )

        except Exception:
            print("Error during load of audio:", self.batch[index]["wav_file"])

        return {
            "wav_path": self.batch[index]["wav_file"],
            "input_values": fbank,
            "label_id": torch.tensor(int(self.label2id[self.batch[index]["label"]])),
            "label": self.batch[index]["label"]
        }


def preprocess_data(data_loader, model, out_path):
    hidden_states_l = []
    wav_paths = []
    labels = []
    labels_id = []

    os.makedirs(out_path, exist_ok=True)

    with torch.no_grad():
        for batch, label_id, wav_path, label in tqdm.tqdm(data_loader):
            batch = batch.to(device)
            outputs = model(
                batch["input_values"],
                batch["attention_mask"],
                output_attentions=True,
                output_hidden_states=True)
            hidden_states = outputs[0]

            # hidden_states_l.append(torch.mean(hidden_states, dim=1))
            wav_paths.append(wav_path)
            labels.append(label)
            labels_id.append(label_id)

            padding_mask = get_feature_vector_attention_mask(hidden_states.shape[1], batch["attention_mask"])
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
            hidden_states_l.append(pooled_output)

    hidden_states_l = functools.reduce(operator.iconcat, hidden_states_l, [])
    wav_paths = functools.reduce(operator.iconcat, wav_paths, [])
    labels = functools.reduce(operator.iconcat, labels, [])
    labels_id = functools.reduce(operator.iconcat, labels_id, [])

    a = []

    for i in labels_id:
        a.append(int(i.cpu().detach().numpy()))

    temp_dict = {"hidden_states": hidden_states_l, "wav_path": wav_paths, "label": labels, "label_id": a}

    dataset = Dataset.from_dict(temp_dict)
    dataset.save_to_disk(out_path)


class DatasetEVO(torch.utils.data.Dataset):
    def __init__(
        self,
        batch
    ):
        self.batch = batch

    def __len__(self):
        return self.batch.num_rows

    def __getitem__(self, index: int) -> Dict[torch.Tensor, np.ndarray]:
        return torch.tensor(self.batch[index]["hidden_states"]), torch.tensor(self.batch[index]["label_id"])


class CustomSupervisedNE(SupervisedNE):
    def _evaluate_network(self, network: nn.Module) -> torch.Tensor:
        loss = 0.0
        for batch_idx in range(self._num_minibatches):
            if not self._common_minibatch:
                self._current_minibatch = self.get_minibatch()
            else:
                self._current_minibatch = self._current_minibatches[batch_idx]
            loss += self._evaluate_using_minibatch(network, self._current_minibatch) / self._num_minibatches
        return loss

