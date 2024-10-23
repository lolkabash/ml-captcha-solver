"""
!pip install tqdm
!pip install tensorflow
!pip install tf2onnx
!pip install onnx
!pip install librosa
!pip install opencv-python
"""

# imports
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model

try:
    [
        tf.config.experimental.set_memory_growth(gpu, True)
        for gpu in tf.config.experimental.list_physical_devices("GPU")
    ]
except:
    pass

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# from mltu.tensorflow.dataProvider import DataProvider
import os
import copy
import typing
import numpy as np
import pandas as pd
import logging
import tf2onnx
import onnx
import importlib
import matplotlib.pyplot as plt
import matplotlib
import cv2
import time
import queue
import threading
import yaml

from tqdm import tqdm
from keras.callbacks import Callback
from keras.metrics import Metric
from . import Image
from enum import Enum
from abc import ABC
from abc import abstractmethod
from PIL import Image as PilImage
from datetime import datetime
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile


class DataProvider:
    def __init__(
        self,
        dataset: typing.Union[str, list, pd.DataFrame],
        data_preprocessors: typing.List[typing.Callable] = None,
        batch_size: int = 4,
        shuffle: bool = True,
        initial_epoch: int = 1,
        augmentors: typing.List[Augmentor] = None,
        transformers: typing.List[Transformer] = None,
        batch_postprocessors: typing.List[typing.Callable] = None,
        skip_validation: bool = True,
        limit: int = None,
        use_cache: bool = False,
        log_level: int = logging.INFO,
        numpy: bool = True,
    ) -> None:
        """Standardised object for providing data to a model while training.

        Attributes:
            dataset (str, list, pd.DataFrame): Path to dataset, list of data or pandas dataframe of data.
            data_preprocessors (list): List of data preprocessors. (e.g. [read image, read audio, etc.])
            batch_size (int): The number of samples to include in each batch. Defaults to 4.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
            initial_epoch (int): The initial epoch. Defaults to 1.
            augmentors (list, optional): List of augmentor functions. Defaults to None.
            transformers (list, optional): List of transformer functions. Defaults to None.
            batch_postprocessors (list, optional): List of batch postprocessor functions. Defaults to None.
            skip_validation (bool, optional): Whether to skip validation. Defaults to True.
            limit (int, optional): Limit the number of samples in the dataset. Defaults to None.
            use_cache (bool, optional): Whether to cache the dataset. Defaults to False.
            log_level (int, optional): The log level. Defaults to logging.INFO.
            numpy (bool, optional): Whether to convert data to numpy. Defaults to True.
        """
        self._dataset = dataset
        self._data_preprocessors = (
            [] if data_preprocessors is None else data_preprocessors
        )
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._epoch = initial_epoch
        self._augmentors = [] if augmentors is None else augmentors
        self._transformers = [] if transformers is None else transformers
        self._batch_postprocessors = (
            [] if batch_postprocessors is None else batch_postprocessors
        )
        self._skip_validation = skip_validation
        self._limit = limit
        self._use_cache = use_cache
        self._step = 0
        self._cache = {}
        self._on_epoch_end_remove = []
        self._numpy = numpy

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Validate dataset
        if not skip_validation:
            self._dataset = self.validate(dataset)
        else:
            self.logger.info("Skipping Dataset validation...")

        # Check if dataset has length
        if not len(dataset):
            raise ValueError("Dataset must be iterable")

        if limit:
            self.logger.info(f"Limiting dataset to {limit} samples.")
            self._dataset = self._dataset[:limit]

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self._dataset) / self._batch_size))

    @property
    def augmentors(self) -> typing.List[Augmentor]:
        """Return augmentors"""
        return self._augmentors

    @augmentors.setter
    def augmentors(self, augmentors: typing.List[Augmentor]):
        """Decorator for adding augmentors to the DataProvider"""
        for augmentor in augmentors:
            if isinstance(augmentor, Augmentor):
                if self._augmentors is not None:
                    self._augmentors.append(augmentor)
                else:
                    self._augmentors = [augmentor]

            else:
                self.logger.warning(
                    f"Augmentor {augmentor} is not an instance of Augmentor."
                )

    @property
    def transformers(self) -> typing.List[Transformer]:
        """Return transformers"""
        return self._transformers

    @transformers.setter
    def transformers(self, transformers: typing.List[Transformer]):
        """Decorator for adding transformers to the DataProvider"""
        for transformer in transformers:
            if isinstance(transformer, Transformer):
                if self._transformers is not None:
                    self._transformers.append(transformer)
                else:
                    self._transformers = [transformer]

            else:
                self.logger.warning(
                    f"Transformer {transformer} is not an instance of Transformer."
                )

    @property
    def epoch(self) -> int:
        """Return Current Epoch"""
        return self._epoch

    @property
    def step(self) -> int:
        """Return Current Step"""
        return self._step

    def on_epoch_end(self):
        """Shuffle training dataset and increment epoch counter at the end of each epoch."""
        self._epoch += 1
        if self._shuffle:
            np.random.shuffle(self._dataset)

        # Remove any samples that were marked for removal
        for remove in self._on_epoch_end_remove:
            self.logger.warning(f"Removing {remove} from dataset.")
            self._dataset.remove(remove)
        self._on_epoch_end_remove = []

    def validate_list_dataset(self, dataset: list) -> list:
        """Validate a list dataset"""
        validated_data = [
            data
            for data in tqdm(dataset, desc="Validating Dataset")
            if os.path.exists(data[0])
        ]
        if not validated_data:
            raise FileNotFoundError("No valid data found in dataset.")

        return validated_data

    def validate(
        self, dataset: typing.Union[str, list, pd.DataFrame]
    ) -> typing.Union[list, str]:
        """Validate the dataset and return the dataset"""

        if isinstance(dataset, str):
            if os.path.exists(dataset):
                return dataset
        elif isinstance(dataset, list):
            return self.validate_list_dataset(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return self.validate_list_dataset(dataset.values.tolist())
        else:
            raise TypeError("Dataset must be a path, list or pandas dataframe.")

    def split(
        self, split: float = 0.9, shuffle: bool = True
    ) -> typing.Tuple[typing.Any, typing.Any]:
        """Split current data provider into training and validation data providers.

        Args:
            split (float, optional): The split ratio. Defaults to 0.9.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

        Returns:
            train_data_provider (tf.keras.utils.Sequence): The training data provider.
            val_data_provider (tf.keras.utils.Sequence): The validation data provider.
        """
        if shuffle:
            np.random.shuffle(self._dataset)

        train_data_provider, val_data_provider = copy.deepcopy(self), copy.deepcopy(
            self
        )
        train_data_provider._dataset = self._dataset[: int(len(self._dataset) * split)]
        val_data_provider._dataset = self._dataset[int(len(self._dataset) * split) :]

        return train_data_provider, val_data_provider

    def to_csv(self, path: str, index: bool = False) -> None:
        """Save the dataset to a csv file

        Args:
            path (str): The path to save the csv file.
            index (bool, optional): Whether to save the index. Defaults to False.
        """
        df = pd.DataFrame(self._dataset)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=index)

    def get_batch_annotations(self, index: int) -> typing.List:
        """Returns a batch of annotations by batch index in the dataset

        Args:
            index (int): The index of the batch in

        Returns:
            batch_annotations (list): A list of batch annotations
        """
        self._step = index
        start_index = index * self._batch_size

        # Get batch indexes
        batch_indexes = [
            i
            for i in range(start_index, start_index + self._batch_size)
            if i < len(self._dataset)
        ]

        # Read batch data
        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations

    def start_executor(self) -> None:
        """Start the executor to process data"""

        def executor(batch_data):
            for data in batch_data:
                yield self.process_data(data)

        if not hasattr(self, "_executor"):
            self._executor = executor

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for index in range(len(self)):
            results = self[index]
            yield results

    def process_data(self, batch_data):
        """Process data batch of data"""
        if (
            self._use_cache
            and batch_data[0] in self._cache
            and isinstance(batch_data[0], str)
        ):
            data, annotation = copy.deepcopy(self._cache[batch_data[0]])
        else:
            data, annotation = batch_data
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)

            if data is None or annotation is None:
                self.logger.warning(
                    "Data or annotation is None, marking for removal on epoch end."
                )
                self._on_epoch_end_remove.append(batch_data)
                return None, None

            if self._use_cache and batch_data[0] not in self._cache:
                self._cache[batch_data[0]] = (
                    copy.deepcopy(data),
                    copy.deepcopy(annotation),
                )

        # Then augment, transform and postprocess the batch data
        for objects in [self._augmentors, self._transformers]:
            for _object in objects:
                data, annotation = _object(data, annotation)

        if self._numpy:
            try:
                data = data.numpy()
                annotation = annotation.numpy()
            except:
                pass

        return data, annotation

    def __getitem__(self, index: int):
        """Returns a batch of processed data by index

        Args:
            index (int): index of batch

        Returns:
            tuple: batch of data and batch of annotations
        """
        if index == 0:
            self.start_executor()

        dataset_batch = self.get_batch_annotations(index)

        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for data, annotation in self._executor(dataset_batch):
            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping.")
                continue
            batch_data.append(data)
            batch_annotations.append(annotation)

        if self._batch_postprocessors:
            for batch_postprocessor in self._batch_postprocessors:
                batch_data, batch_annotations = batch_postprocessor(
                    batch_data, batch_annotations
                )

            return batch_data, batch_annotations

        try:
            return np.array(batch_data), np.array(batch_annotations)
        except:
            return batch_data, batch_annotations


# from mltu.tensorflow.losses import CTCloss


class CTCloss(tf.keras.losses.Loss):
    """CTCLoss objec for training the model"""

    def __init__(self, name: str = "CTCloss") -> None:
        super(CTCloss, self).__init__()
        self.name = name
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def __call__(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None
    ) -> tf.Tensor:
        """Compute the training batch CTC loss value"""
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        return loss


# from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
class Model2onnx(Callback):
    """Converts the model to onnx format after training is finished."""

    def __init__(
        self,
        saved_model_path: str,
        metadata: dict = None,
        save_on_epoch_end: bool = False,
    ) -> None:
        """Converts the model to onnx format after training is finished.
        Args:
            saved_model_path (str): Path to the saved .h5 model.
            metadata (dict, optional): Dictionary containing metadata to be added to the onnx model. Defaults to None.
            save_on_epoch_end (bool, optional): Save the onnx model on every epoch end. Defaults to False.
        """
        super().__init__()
        self.saved_model_path = saved_model_path
        self.metadata = metadata
        self.save_on_epoch_end = save_on_epoch_end

        try:
            import tf2onnx
        except:
            raise ImportError(
                "tf2onnx is not installed. Please install it using 'pip install tf2onnx'"
            )

        try:
            import onnx
        except:
            raise ImportError(
                "onnx is not installed. Please install it using 'pip install onnx'"
            )

    @staticmethod
    def model2onnx(model: tf.keras.Model, onnx_model_path: str):
        try:
            import tf2onnx

            # convert the model to onnx format
            tf2onnx.convert.from_keras(model, output_path=onnx_model_path)

        except Exception as e:
            print(e)

    @staticmethod
    def include_metadata(onnx_model_path: str, metadata: dict = None):
        try:
            if metadata and isinstance(metadata, dict):

                import onnx

                # Load the ONNX model
                onnx_model = onnx.load(onnx_model_path)

                # Add the metadata dictionary to the model's metadata_props attribute
                for key, value in metadata.items():
                    meta = onnx_model.metadata_props.add()
                    meta.key = key
                    meta.value = str(value)

                # Save the modified ONNX model
                onnx.save(onnx_model, onnx_model_path)

        except Exception as e:
            print(e)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        """Converts the model to onnx format on every epoch end."""
        if self.save_on_epoch_end:
            self.on_train_end(logs=logs)

    def on_train_end(self, logs=None):
        """Converts the model to onnx format after training is finished."""
        self.model.load_weights(self.saved_model_path)
        onnx_model_path = self.saved_model_path.replace(".h5", ".onnx")
        self.model2onnx(self.model, onnx_model_path)
        self.include_metadata(onnx_model_path, self.metadata)


class TrainLogger(Callback):
    """Logs training metrics to a file.

    Args:
        log_path (str): Path to the directory where the log file will be saved.
        log_file (str, optional): Name of the log file. Defaults to 'logs.log'.
        logLevel (int, optional): Logging level. Defaults to logging.INFO.
    """

    def __init__(
        self,
        log_path: str,
        log_file: str = "logs.log",
        logLevel=logging.INFO,
        console_output=False,
    ) -> None:
        super().__init__()
        self.log_path = log_path
        self.log_file = log_file

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.logger = logging.getLogger()
        self.logger.setLevel(logLevel)

        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        self.file_handler = logging.FileHandler(
            os.path.join(self.log_path, self.log_file)
        )
        self.file_handler.setLevel(logLevel)
        self.file_handler.setFormatter(self.formatter)

        if not console_output:
            self.logger.handlers[:] = []

        self.logger.addHandler(self.file_handler)

    def on_epoch_end(self, epoch: int, logs: dict = None):
        epoch_message = f"Epoch {epoch}; "
        logs_message = "; ".join([f"{key}: {value}" for key, value in logs.items()])
        self.logger.info(epoch_message + logs_message)


class WarmupCosineDecay(Callback):
    """Cosine decay learning rate scheduler with warmup

    Args:
        lr_after_warmup (float): Learning rate after warmup
        final_lr (float): Final learning rate
        warmup_epochs (int): Number of warmup epochs
        decay_epochs (int): Number of decay epochs
        initial_lr (float, optional): Initial learning rate. Defaults to 0.0.
        verbose (bool, optional): Whether to print learning rate. Defaults to False.
    """

    def __init__(
        self,
        lr_after_warmup: float,
        final_lr: float,
        warmup_epochs: int,
        decay_epochs: int,
        initial_lr: float = 0.0,
        verbose=False,
    ) -> None:
        super(WarmupCosineDecay, self).__init__()
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.initial_lr = initial_lr
        self.verbose = verbose

    def on_epoch_begin(self, epoch: int, logs: dict = None):
        """Adjust learning rate at the beginning of each epoch"""

        if epoch >= self.warmup_epochs + self.decay_epochs:
            return logs

        if epoch < self.warmup_epochs:
            lr = (
                self.initial_lr
                + (self.lr_after_warmup - self.initial_lr)
                * (epoch + 1)
                / self.warmup_epochs
            )
        else:
            progress = (epoch - self.warmup_epochs) / self.decay_epochs
            lr = self.final_lr + 0.5 * (self.lr_after_warmup - self.final_lr) * (
                1 + tf.cos(tf.constant(progress) * 3.14159)
            )

        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

        if self.verbose:
            print(f"Epoch {epoch + 1} - Learning Rate: {lr}")

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}

        # Log the learning rate value
        logs["lr"] = self.model.optimizer.lr

        return logs


# from mltu.tensorflow.metrics import CWERMetric
class CWERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Character Error Rate (CER).

    Args:
        padding_token: An integer representing the padding token in the input data.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, padding_token, name="CWER", **kwargs):
        # Initialize the base Metric class
        super(CWERMetric, self).__init__(name=name, **kwargs)

        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.cer_accumulator = tf.Variable(
            0.0, name="cer_accumulator", dtype=tf.float32
        )
        self.wer_accumulator = tf.Variable(
            0.0, name="wer_accumulator", dtype=tf.float32
        )
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)

        # Store the padding token as an attribute
        self.padding_token = padding_token

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables of the metric.

        Args:
            y_true: A tensor of true labels with shape (batch_size, sequence_length).
            y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
            sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(
            input_shape[1], "int32"
        )

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(
            y_pred, input_length, greedy=True
        )

        # Convert the dense decode tensor to a sparse tensor
        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(
            decode_predicted[0], input_length
        )

        # Convert the dense true labels tensor to a sparse tensor and cast to int64
        true_labels_sparse = tf.cast(
            tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64"
        )

        # Retain only the non-padding elements in the predicted labels tensor
        predicted_labels_sparse = tf.sparse.retain(
            predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1)
        )

        # Retain only the non-padding elements in the true labels tensor
        true_labels_sparse = tf.sparse.retain(
            true_labels_sparse,
            tf.not_equal(true_labels_sparse.values, self.padding_token),
        )

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = tf.edit_distance(
            predicted_labels_sparse, true_labels_sparse, normalize=True
        )

        # Add the sum of the distance tensor to the cer_accumulator variable
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))

        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(input_shape[0])

        # Calculate the number of wrong words in batch and add to wer_accumulator variable
        self.wer_accumulator.assign_add(
            tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))
        )

    def result(self):
        """Computes and returns the metric result.

        Returns:
            A dictionary containing the CER and WER.
        """
        return {
            "CER": tf.math.divide_no_nan(
                self.cer_accumulator, tf.cast(self.batch_counter, tf.float32)
            ),
            "WER": tf.math.divide_no_nan(
                self.wer_accumulator, tf.cast(self.batch_counter, tf.float32)
            ),
        }


class CERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Character Error Rate (CER).

    Args:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, vocabulary, name="CER", **kwargs):
        # Initialize the base Metric class
        super(CERMetric, self).__init__(name=name, **kwargs)

        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.cer_accumulator = tf.Variable(
            0.0, name="cer_accumulator", dtype=tf.float32
        )
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)

        # Store the vocabulary as an attribute
        self.vocabulary = tf.constant(list(vocabulary))

    @staticmethod
    def get_cer(pred_decoded, y_true, vocab, padding=-1):
        """Calculates the character error rate (CER) between the predicted labels and true labels for a batch of input data.

        Args:
            pred_decoded (tf.Tensor): The predicted labels, with dtype=tf.int32, usually output from tf.keras.backend.ctc_decode
            y_true (tf.Tensor): The true labels, with dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, with dtype=tf.string
            padding (int, optional): The padding token when converting to sparse tensor. Defaults to -1.

        Returns:
            tf.Tensor: The CER between the predicted labels and true labels
        """
        # Keep only valid indices in the predicted labels tensor, replacing invalid indices with padding token
        vocab_length = tf.cast(tf.shape(vocab)[0], tf.int64)
        valid_pred_indices = tf.less(pred_decoded, vocab_length)
        valid_pred = tf.where(valid_pred_indices, pred_decoded, padding)

        # Keep only valid indices in the true labels tensor, replacing invalid indices with padding token
        y_true = tf.cast(y_true, tf.int64)
        valid_true_indices = tf.less(y_true, vocab_length)
        valid_true = tf.where(valid_true_indices, y_true, padding)

        # Convert the valid predicted labels tensor to a sparse tensor
        sparse_pred = tf.RaggedTensor.from_tensor(
            valid_pred, padding=padding
        ).to_sparse()

        # Convert the valid true labels tensor to a sparse tensor
        sparse_true = tf.RaggedTensor.from_tensor(
            valid_true, padding=padding
        ).to_sparse()

        # Calculate the normalized edit distance between the sparse predicted labels tensor and sparse true labels tensor
        distance = tf.edit_distance(sparse_pred, sparse_true, normalize=True)

        return distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables of the metric.

        Args:
            y_true: A tensor of true labels with shape (batch_size, sequence_length).
            y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
            sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(
            input_shape[1], "int32"
        )

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(
            y_pred, input_length, greedy=True
        )

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = self.get_cer(decode_predicted[0], y_true, self.vocabulary)

        # Add the sum of the distance tensor to the cer_accumulator variable
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))

        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(input_shape[0])

    def result(self):
        """Computes and returns the metric result.

        Returns:
            A TensorFlow float representing the CER (character error rate).
        """
        return tf.math.divide_no_nan(
            self.cer_accumulator, tf.cast(self.batch_counter, tf.float32)
        )


class WERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Word Error Rate (WER).

    Attributes:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, vocabulary: str, name="WER", **kwargs):
        # Initialize the base Metric class
        super(WERMetric, self).__init__(name=name, **kwargs)

        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.wer_accumulator = tf.Variable(
            0.0, name="wer_accumulator", dtype=tf.float32
        )
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)

        # Store the vocabulary as an attribute
        self.vocabulary = tf.constant(list(vocabulary))

    @staticmethod
    def preprocess_dense(
        dense_input: tf.Tensor, vocab: tf.Tensor, padding=-1, separator=""
    ) -> tf.SparseTensor:
        """Preprocess the dense input tensor to a sparse tensor with given vocabulary

        Args:
            dense_input (tf.Tensor): The dense input tensor, dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, dtype=tf.string
            padding (int, optional): The padding token when converting to sparse tensor. Defaults to -1.

        Returns:
            tf.SparseTensor: The sparse tensor with given vocabulary
        """
        # Keep only the valid indices of the dense input tensor
        vocab_length = tf.cast(tf.shape(vocab)[0], tf.int64)
        dense_input = tf.cast(dense_input, tf.int64)
        valid_indices = tf.less(dense_input, vocab_length)
        valid_input = tf.where(valid_indices, dense_input, padding)

        # Convert the valid input tensor to a ragged tensor with padding
        input_ragged = tf.RaggedTensor.from_tensor(valid_input, padding=padding)

        # Use the vocabulary tensor to get the strings corresponding to the indices in the ragged tensor
        input_binary_chars = tf.gather(vocab, input_ragged)

        # Join the binary character tensor along the sequence axis to get the input strings
        input_strings = tf.strings.reduce_join(
            input_binary_chars, axis=1, separator=separator
        )

        # Convert the input strings tensor to a sparse tensor
        input_sparse_string = tf.strings.split(input_strings, sep=" ").to_sparse()

        return input_sparse_string

    @staticmethod
    def get_wer(pred_decoded, y_true, vocab, padding=-1, separator=""):
        """Calculate the normalized WER distance between the predicted labels and true labels tensors

        Args:
            pred_decoded (tf.Tensor): The predicted labels tensor, dtype=tf.int32. Usually output from tf.keras.backend.ctc_decode
            y_true (tf.Tensor): The true labels tensor, dtype=tf.int32
            vocab (tf.Tensor): The vocabulary tensor, dtype=tf.string

        Returns:
            tf.Tensor: The normalized WER distance between the predicted labels and true labels tensors
        """
        pred_sparse = WERMetric.preprocess_dense(
            pred_decoded, vocab, padding=padding, separator=separator
        )
        true_sparse = WERMetric.preprocess_dense(
            y_true, vocab, padding=padding, separator=separator
        )

        distance = tf.edit_distance(pred_sparse, true_sparse, normalize=True)

        # test with numerical labels not string
        # true_sparse = tf.RaggedTensor.from_tensor(y_true, padding=-1).to_sparse()

        # replace 23 with -1
        # pred_decoded2 = tf.where(tf.equal(pred_decoded, 23), -1, pred_decoded)
        # pred_decoded2_sparse = tf.RaggedTensor.from_tensor(pred_decoded2, padding=-1).to_sparse()

        # distance = tf.edit_distance(pred_decoded2_sparse, true_sparse, normalize=True)

        return distance

    def update_state(self, y_true, y_pred, sample_weight=None):
        """ """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype="int32") * tf.cast(
            input_shape[1], "int32"
        )

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(
            y_pred, input_length, greedy=True
        )

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = self.get_wer(decode_predicted[0], y_true, self.vocabulary)

        # Calculate the number of wrong words in batch and add to wer_accumulator variable
        self.wer_accumulator.assign_add(tf.reduce_sum(tf.cast(distance, tf.float32)))

        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(input_shape[0])

    def result(self):
        """Computes and returns the metric result.

        Returns:
            A TensorFlow float representing the WER (Word Error Rate).
        """
        return tf.math.divide_no_nan(
            self.wer_accumulator, tf.cast(self.batch_counter, tf.float32)
        )


# from mltu.preprocessors import ImageReader


# from mltu.annotations.audio import Audio
class Audio:
    """Audio object

    Attributes:
        audio (np.ndarray): Audio array
        sample_rate (int): Sample rate
        init_successful (bool): True if audio was successfully read
        library (object): Library used to read audio, tested only with librosa
    """

    init_successful = False
    augmented = False

    def __init__(self, audioPath: str, sample_rate: int = 22050, library=None) -> None:
        if library is None:
            raise ValueError("library must be provided. (e.g. librosa object)")

        if isinstance(audioPath, str):
            if not os.path.exists(audioPath):
                raise FileNotFoundError(f"Image {audioPath} not found.")

            self._audio, self.sample_rate = library.load(audioPath, sr=sample_rate)
            self.path = audioPath
            self.init_successful = True

        else:
            raise TypeError(
                f"audioPath must be path to audio file, not {type(audioPath)}"
            )

    @property
    def audio(self) -> np.ndarray:
        return self._audio

    @audio.setter
    def audio(self, value: np.ndarray):
        self.augmented = True
        self._audio = value

    @property
    def shape(self) -> tuple:
        return self._audio.shape

    def numpy(self) -> np.ndarray:
        return self._audio

    def __add__(self, other: np.ndarray) -> np.ndarray:
        self._audio = self._audio + other
        self.augmented = True
        return self

    def __len__(self) -> int:
        return len(self._audio)

    def __call__(self) -> np.ndarray:
        return self._audio

    def __repr__(self):
        return repr(self._audio)

    def __array__(self):
        return self._audio


""" Implemented Preprocessors:
- ImageReader - Read image from path and return image and label
- AudioReader - Read audio from path and return audio and label
- WavReader - Read wav file with librosa and return spectrogram and label
- ImageCropper - Crop image to (width, height)
"""


class ImageReader:
    """Read image from path and return image and label"""

    def __init__(
        self,
        image_class: Image,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._image_class = image_class

    def __call__(
        self, image_path: typing.Union[str, np.ndarray], label: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Read image from path and return image and label

        Args:
            image_path (typing.Union[str, np.ndarray]): Path to image or numpy array
            label (Any): Label of image

        Returns:
            Image: Image object
            Any: Label of image
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
        elif isinstance(image_path, np.ndarray):
            pass
        else:
            raise TypeError(f"Image {image_path} is not a string or numpy array.")

        image = self._image_class(image=image_path)

        if not image.init_successful:
            image = None
            self.logger.warning(
                f"Image {image_path} could not be read, returning None."
            )

        return image, label


def import_librosa(object) -> None:
    """Import librosa using importlib"""
    try:
        version = object.librosa.__version__
    except:
        version = "librosa version not found"
        try:
            object.librosa = importlib.import_module("librosa")
            print("librosa version:", object.librosa.__version__)
        except:
            raise ImportError(
                "librosa is required to augment Audio. Please install it with `pip install librosa`."
            )


class AudioReader:
    """Read audio from path and return audio and label

    Attributes:
        sample_rate (int): Sample rate. Defaults to None.
        log_level (int): Log level. Defaults to logging.INFO.
    """

    def __init__(
        self,
        sample_rate=None,
        log_level: int = logging.INFO,
    ) -> None:
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module("librosa")
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError(
                "librosa is required to augment Audio. Please install it with `pip install librosa`."
            )

    def __call__(
        self, audio_path: str, label: typing.Any
    ) -> typing.Tuple[np.ndarray, typing.Any]:
        """Read audio from path and return audio and label

        Args:
            audio_path (str): Path to audio
            label (Any): Label of audio

        Returns:
            Audio: Audio object
            Any: Label of audio
        """
        if isinstance(audio_path, str):
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio {audio_path} not found.")
        else:
            raise TypeError(f"Audio {audio_path} is not a string.")

        audio = Audio(audio_path, sample_rate=self.sample_rate, library=self.librosa)

        if not audio.init_successful:
            audio = None
            self.logger.warning(
                f"Audio {audio_path} could not be read, returning None."
            )

        return audio, label


class WavReader:
    """Read wav file with librosa and return audio and label

    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
    """

    def __init__(
        self,
        frame_length: int = 256,
        frame_step: int = 160,
        fft_length: int = 384,
        *args,
        **kwargs,
    ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        matplotlib.interactive(False)
        # import librosa using importlib
        import_librosa(self)

    @staticmethod
    def get_spectrogram(
        wav_path: str, frame_length: int, frame_step: int, fft_length: int
    ) -> np.ndarray:
        """Compute the spectrogram of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            frame_length (int): Length of the frames in samples.
            frame_step (int): Step size between frames in samples.
            fft_length (int): Number of FFT components.

        Returns:
            np.ndarray: Spectrogram of the WAV file.
        """
        import_librosa(WavReader)

        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = WavReader.librosa.load(wav_path)

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = WavReader.librosa.stft(
            audio, hop_length=frame_step, win_length=frame_length, n_fft=fft_length
        ).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (
            np.std(spectrogram) + 1e-10
        )

        return spectrogram

    @staticmethod
    def plot_raw_audio(wav_path: str, title: str = None, sr: int = 16000) -> None:
        """Plot the raw audio of a WAV file

        Args:
            wav_path (str): Path to the WAV file.
            sr (int, optional): Sample rate of the WAV file. Defaults to 16000.
            title (str, optional): Title
        """
        import_librosa(WavReader)
        # Load the wav file and store the audio data in the variable 'audio' and the sample rate in 'orig_sr'
        audio, orig_sr = WavReader.librosa.load(wav_path, sr=sr)

        duration = len(audio) / orig_sr

        time = np.linspace(0, duration, num=len(audio))

        plt.figure(figsize=(15, 5))
        plt.plot(time, audio)
        plt.title(title) if title else plt.title("Audio Plot")
        plt.ylabel("signal wave")
        plt.xlabel("time (s)")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_spectrogram(
        spectrogram: np.ndarray,
        title: str = "",
        transpose: bool = True,
        invert: bool = True,
    ) -> None:
        """Plot the spectrogram of a WAV file

        Args:
            spectrogram (np.ndarray): Spectrogram of the WAV file.
            title (str, optional): Title of the plot. Defaults to None.
            transpose (bool, optional): Transpose the spectrogram. Defaults to True.
            invert (bool, optional): Invert the spectrogram. Defaults to True.
        """
        if transpose:
            spectrogram = spectrogram.T

        if invert:
            spectrogram = spectrogram[::-1]

        plt.figure(figsize=(15, 5))
        plt.imshow(spectrogram, aspect="auto", origin="lower")
        plt.title(f"Spectrogram: {title}")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    def __call__(self, audio_path: str, label: typing.Any):
        """
        Extract the spectrogram and label of a WAV file.

        Args:
            audio_path (str): Path to the WAV file.
            label (typing.Any): Label of the WAV file.

        Returns:
            Tuple[np.ndarray, typing.Any]: Spectrogram of the WAV file and its label.
        """
        return (
            self.get_spectrogram(
                audio_path, self.frame_length, self.frame_step, self.fft_length
            ),
            label,
        )


class ImageCropper:
    """Crop image to (width, height)

    Attributes:
        width (int): Width of image
        height (int): Height of image
        wifth_offset (int): Offset for width
        height_offset (int): Offset for height
    """

    def __init__(
        self,
        width: int,
        height: int,
        width_offset: int = 0,
        height_offset: int = 0,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        self._width = width
        self._height = height
        self._width_offset = width_offset
        self._height_offset = height_offset

    def __call__(
        self, image: Image, label: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        image_numpy = image.numpy()

        source_width, source_height = image_numpy.shape[:2][::-1]

        if source_width >= self._width:
            image_numpy = image_numpy[
                :, self._width_offset : self._width + self._width_offset
            ]
        else:
            raise Exception("unexpected")

        if source_height >= self._height:
            image_numpy = image_numpy[
                self._height_offset : self._height + self._height_offset, :
            ]
        else:
            raise Exception("unexpected")

        image.update(image_numpy)

        return image, label


# from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding


# from mltu.annotations.audio import Audio
# from mltu.annotations.detections import Detections
class BboxType(Enum):
    XYWH = 1  # x center, y center, width, height
    XYXY = 2  # left, top, right, bottom
    LTWH = 3  # left, top, width, height


class Detection:
    """Object to hold the information of a detection for simple manipulation and visualization"""

    def __init__(
        self,
        bbox,
        label: str,
        labels: dict = {},
        bbox_type: BboxType = BboxType.XYWH,
        confidence: float = 0.0,
        image_path: str = "",
        width: int = None,
        height: int = None,
        relative: bool = False,
    ):
        """
        Args:
            bbox (list or np.ndarray): Bounding box coordinates
            label (str): Label of the detection
            labels (dict, optional): Dictionary of labels. Defaults to {}.
            bbox_type (BboxType, optional): Type of the bounding box coordinates. Defaults to BboxType.XYWH.
            confidence (float, optional): Confidence score of the detection. Defaults to 0.0.
            image_path (str, optional): Path to the image. Defaults to "".
            width (int, optional): Width of the image. Defaults to None.
            height (int, optional): Height of the image. Defaults to None.
            relative (bool, optional): Whether the bounding box coordinates are relative to the image size. Defaults to False.
        """
        self.bbox = np.array(bbox)
        self.label = label
        self.labels = labels
        self.bbox_type = bbox_type
        self.confidence = confidence
        self.image_path = image_path
        self.width = width
        self.height = height
        self.relative = relative

        self.augmented = False

        self._xywh = None
        self._xyxy = None

        self.validate()

    @property
    def labelId(self) -> int:
        return self.label2id(self.label)

    def label2id(self, label: str) -> int:
        labelId = {v: k for k, v in self.labels.items()}.get(label, None)
        if labelId is None:
            raise ValueError(f"label {label} not found in labels")

        return labelId

    @property
    def xywh(self):
        return self._xywh

    @xywh.setter
    def xywh(self, xywh: np.ndarray):
        if (xywh[:2] + xywh[2:] / 2 > 1).any():
            # fix the bbox to be in range [0, 1]
            self._xywh = self.xyxy2xywh(self.xywh2xyxy(xywh))
        else:
            self._xywh = xywh.clip(0, 1)

        self._xyxy = self.xywh2xyxy(self._xywh)

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def xyxy_abs(self):
        return (
            self.xyxy * np.array([self.width, self.height, self.width, self.height])
        ).astype(int)

    @staticmethod
    def xywh2xyxy(xywh: np.ndarray):
        """Convert bounding box from x, y, width, height to x1, y1, x2, y2"""
        x, y, w, h = xywh
        x, y = x - w / 2, y - h / 2
        return np.array([x, y, x + w, y + h]).clip(0, 1)

    @staticmethod
    def xyxy2xywh(xyxy: np.ndarray):
        """Convert bounding box from x1, y1, x2, y2 to x, y, width, height"""
        x, y, x2, y2 = xyxy
        w, h = x2 - x, y2 - y
        return np.array([x + w / 2, y + h / 2, w, h]).clip(0, 1)

    @staticmethod
    def ltwh2xywh(ltwh: np.ndarray):
        """Convert bounding box from left, top, width, height to x, y, width, height"""
        l, t, w, h = ltwh
        return np.array([l + w / 2, t + h / 2, w, h]).clip(0, 1)

    def validate(self):
        """Validate the bounding box coordinates"""
        assert self.bbox_type in BboxType, f"bbox_type must be one of {BboxType}"
        if not self.relative:
            if self.width is None or self.height is None:
                raise ValueError(
                    "width and height must be provided when relative is False"
                )

            if not (np.array(self.bbox) > 1.0).any():
                raise ValueError(
                    "bbox coordinates must be in range [0, np.inf] when relative is False"
                )

            bbox = np.array(self.bbox) / np.array(
                [self.width, self.height, self.width, self.height]
            )

        else:
            bbox = self.bbox

        if self.bbox_type.name == "XYWH":
            self.xywh = bbox

        elif self.bbox_type.name == "XYXY":
            self.xywh = self.xyxy2xywh(bbox)

        elif self.bbox_type.name == "LTWH":
            self.xywh = self.ltwh2xywh(bbox)

        else:
            raise ValueError(f"bbox_type {self.bbox_type} not supported")

    def flip(self, direction: int):
        new_xywh = self.xywh
        if direction == 0:  # mirror
            new_xywh[0] = 1 - new_xywh[0]

        elif direction == 1:  # vertical
            new_xywh[1] = 1 - new_xywh[1]

        self.xywh = new_xywh

        self.augmented = True

    def dot(self, rotMat: np.ndarray, width: int, height: int):
        """Apply transformation matrix to detection

        Args:
            matrix (np.ndarray): Transformation matrix
            width (int): Width of the image
            height (int): Height of the image

        Returns:
            Object with transformed coordinates
        """
        # get the four corners of the bounding box
        bb = np.array(self.xyxy) * np.array(
            [self.width, self.height, self.width, self.height]
        )
        bb = np.array(((bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])))

        bb_rotated = np.vstack(
            (bb.T, np.array((1, 1, 1, 1)))
        )  # Convert the array to [x,y,1] format to dot it with the rotMat
        bb_rotated = np.dot(
            rotMat, bb_rotated
        ).T  # Perform Dot product and get back the points in shape of (4,2)

        # get the new coordinates of the bounding box
        x_min = min(bb_rotated[:, 0])
        y_min = min(bb_rotated[:, 1])
        x_max = max(bb_rotated[:, 0])
        y_max = max(bb_rotated[:, 1])

        new_x = (x_min + x_max) / 2
        new_y = (y_min + y_max) / 2
        new_w = x_max - x_min
        new_h = y_max - y_min

        # Normalize to the new width and height
        new_x /= width
        new_y /= height
        new_w /= width
        new_h /= height

        self.xywh = np.array([new_x, new_y, new_w, new_h])

        self.width = width
        self.height = height
        self.augmented = True

        return self

    def applyToFrame(
        self,
        frame: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        **kwargs,
    ) -> np.ndarray:
        """Draw the bounding box on the image"""
        # Get the coordinates of the bounding box
        x, y, x2, y2 = (
            self.xyxy * np.array([self.width, self.height, self.width, self.height])
        ).astype(np.int32)

        # Draw the bounding box on the image
        frame = cv2.rectangle(
            frame.copy(), (x, y), (x2, y2), color, thickness, **kwargs
        )

        label = (
            f"{self.label}: {self.confidence:.2f}"
            if self.confidence > 0
            else self.label
        )

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        label_y = y - 10 if y - 10 > label_height else y + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            frame,
            (x, label_y - label_height),
            (x + label_width, label_y + label_height),
            color,
            cv2.FILLED,
        )

        cv2.putText(
            frame,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        return frame

    def json(self):
        return {
            "xywh": self.xywh.tolist(),
            "label": self.label,
            "confidence": self.confidence,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
        }


class Detections:
    """Object to hold the information of multiple detections for simple manipulation and visualization"""

    def __init__(
        self,
        labels: dict,
        width: int,
        height: int,
        detections: typing.Iterable[Detection] = [],
        image_path: str = "",
        color_palette: list = [],
    ) -> None:
        """Initialize the Detections object

        Args:
            labels (dict): Dictionary of labels
            width (int): Width of the image
            height (int): Height of the image
            detections (typing.Iterable[Detection], optional): List of detections. Defaults to [].
            image_path (str, optional): Path to the image. Defaults to "".
            color_palette (list, optional): List of colors to use for the bounding boxes. Defaults to [].
        """
        self.labels = labels
        self.width = width
        self.height = height
        self.detections = detections
        self.image_path = image_path
        self.color_palette = color_palette

        self.validate()

    def label2id(self, label: str) -> int:
        labelId = {v: k for k, v in self.labels.items()}.get(label, None)
        if labelId is None:
            raise ValueError(f"label {label} not found in labels")

        return labelId

    def validate(self):
        for detection in self.detections:
            if not isinstance(detection, Detection):
                raise TypeError(
                    f"detections must be iterable of Detection, not {type(detection)}"
                )

            detection.width = self.width
            detection.height = self.height
            detection.labels = self.labels
            detection.image_path = self.image_path

        if isinstance(self.labels, list):
            self.labels = {i: label for i, label in enumerate(self.labels)}

        if not self.labels:
            self.labels = {
                k: v
                for k, v in enumerate(
                    sorted(set([detection.label for detection in self.detections]))
                )
            }

    def applyToFrame(self, image: np.ndarray, **kwargs: dict) -> np.ndarray:
        """Draw the detections on the image"""
        for detection in self.detections:
            color = (
                self.color_palette[detection.labelId]
                if len(self.color_palette) == len(self.labels)
                else (0, 255, 0)
            )
            image = detection.applyToFrame(image, color=color, **kwargs)

        return image

    def __iter__(self):
        return iter(self.detections)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, index: int):
        return self.detections[index]


""" Implemented Transformers:
- ExpandDims - Expand dimension of data
- ImageResizer - Resize image to (width, height)
- LabelIndexer - Convert label to index by vocab
- LabelPadding - Pad label to max_word_length
- ImageNormalizer - Normalize image to float value, transpose axis if necessary and convert to numpy
- SpectrogramPadding - Pad spectrogram to max_spectrogram_length
- AudioToSpectrogram - Convert Audio to Spectrogram
- ImageShowCV2 - Show image for visual inspection
"""


class Transformer:
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError


class ExpandDims(Transformer):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, self.axis), label


class ImageResizer(Transformer):
    """Resize image to (width, height)

    Attributes:
        width (int): Width of image
        height (int): Height of image
        keep_aspect_ratio (bool): Whether to keep aspect ratio of image
        padding_color (typing.Tuple[int]): Color to pad image
    """

    def __init__(
        self,
        width: int,
        height: int,
        keep_aspect_ratio: bool = False,
        padding_color: typing.Tuple[int] = (0, 0, 0),
    ) -> None:
        self._width = width
        self._height = height
        self._keep_aspect_ratio = keep_aspect_ratio
        self._padding_color = padding_color

    @staticmethod
    def unpad_maintaining_aspect_ratio(
        padded_image: np.ndarray, original_width: int, original_height: int
    ) -> np.ndarray:
        height, width = padded_image.shape[:2]
        ratio = min(width / original_width, height / original_height)

        delta_w = width - int(original_width * ratio)
        delta_h = height - int(original_height * ratio)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        unpaded_image = padded_image[top : height - bottom, left : width - right]

        original_image = cv2.resize(unpaded_image, (original_width, original_height))

        return original_image

    @staticmethod
    def resize_maintaining_aspect_ratio(
        image: np.ndarray,
        width_target: int,
        height_target: int,
        padding_color: typing.Tuple[int] = (0, 0, 0),
    ) -> np.ndarray:
        """Resize image maintaining aspect ratio and pad with padding_color.

        Args:
            image (np.ndarray): Image to resize
            width_target (int): Target width
            height_target (int): Target height
            padding_color (typing.Tuple[int]): Color to pad image

        Returns:
            np.ndarray: Resized image
        """
        height, width = image.shape[:2]
        ratio = min(width_target / width, height_target / height)
        new_w, new_h = int(width * ratio), int(height * ratio)

        resized_image = cv2.resize(image, (new_w, new_h))
        delta_w = width_target - new_w
        delta_h = height_target - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_image = cv2.copyMakeBorder(
            resized_image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=padding_color,
        )

        return new_image

    def __call__(
        self, image: Image, label: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        if not isinstance(image, Image):
            raise TypeError(f"Expected image to be of type Image, got {type(image)}")

        # Maintains aspect ratio and resizes with padding.
        if self._keep_aspect_ratio:
            image_numpy = self.resize_maintaining_aspect_ratio(
                image.numpy(), self._width, self._height, self._padding_color
            )
            if isinstance(label, Image):
                label_numpy = self.resize_maintaining_aspect_ratio(
                    label.numpy(), self._width, self._height, self._padding_color
                )
                label.update(label_numpy)
        else:
            # Resizes without maintaining aspect ratio.
            image_numpy = cv2.resize(image.numpy(), (self._width, self._height))
            if isinstance(label, Image):
                label_numpy = cv2.resize(label.numpy(), (self._width, self._height))
                label.update(label_numpy)

        image.update(image_numpy)

        return image, label


class LabelIndexer(Transformer):
    """Convert label to index by vocab

    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """

    def __init__(self, vocab: typing.List[str]) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])


class LabelPadding(Transformer):
    """Pad label to max_word_length

    Attributes:
        padding_value (int): Value to pad
        max_word_length (int): Maximum length of label
        use_on_batch (bool): Whether to use on batch. Default: False
    """

    def __init__(
        self,
        padding_value: int,
        max_word_length: int = None,
        use_on_batch: bool = False,
    ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_word_length is None:
            raise ValueError(
                "max_word_length must be specified if use_on_batch is False"
            )

    def __call__(self, data: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in label])
            padded_labels = []
            for l in label:
                padded_label = np.pad(
                    l,
                    (0, max_len - len(l)),
                    "constant",
                    constant_values=self.padding_value,
                )
                padded_labels.append(padded_label)

            padded_labels = np.array(padded_labels)
            return data, padded_labels

        label = label[: self.max_word_length]
        return data, np.pad(
            label,
            (0, self.max_word_length - len(label)),
            "constant",
            constant_values=self.padding_value,
        )


class ImageNormalizer:
    """Normalize image to float value, transpose axis if necessary and convert to numpy"""

    def __init__(self, transpose_axis: bool = False):
        """Initialize ImageNormalizer

        Args:
            transpose_axis (bool): Whether to transpose axis. Default: False
        """
        self.transpose_axis = transpose_axis

    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[np.ndarray, typing.Any]:
        """Convert each Image to numpy, transpose axis ant normalize to float value"""
        img = image.numpy() / 255.0

        if self.transpose_axis:
            img = img.transpose(2, 0, 1)

        return img, annotation


class SpectrogramPadding(Transformer):
    """Pad spectrogram to max_spectrogram_length

    Attributes:
        padding_value (int): Value to pad
        max_spectrogram_length (int): Maximum length of spectrogram. Must be specified if use_on_batch is False. Default: None
        use_on_batch (bool): Whether to use on batch. Default: False
    """

    def __init__(
        self,
        padding_value: int,
        max_spectrogram_length: int = None,
        use_on_batch: bool = False,
    ) -> None:
        self.max_spectrogram_length = max_spectrogram_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_spectrogram_length is None:
            raise ValueError(
                "max_spectrogram_length must be specified if use_on_batch is False"
            )

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in spectrogram])
            padded_spectrograms = []
            for spec in spectrogram:
                padded_spectrogram = np.pad(
                    spec,
                    ((0, max_len - spec.shape[0]), (0, 0)),
                    mode="constant",
                    constant_values=self.padding_value,
                )
                padded_spectrograms.append(padded_spectrogram)

            padded_spectrograms = np.array(padded_spectrograms)
            label = np.array(label)

            return padded_spectrograms, label

        padded_spectrogram = np.pad(
            spectrogram,
            ((0, self.max_spectrogram_length - spectrogram.shape[0]), (0, 0)),
            mode="constant",
            constant_values=self.padding_value,
        )

        return padded_spectrogram, label


class AudioPadding(Transformer):
    def __init__(
        self,
        max_audio_length: int,
        padding_value: int = 0,
        use_on_batch: bool = False,
        limit: bool = False,
    ):
        super(AudioPadding, self).__init__()
        self.max_audio_length = max_audio_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch
        self.limit = limit

    def __call__(self, audio: Audio, label: typing.Any):
        # batched padding
        if self.use_on_batch:
            max_len = max([len(a) for a in audio])
            padded_audios = []
            for a in audio:
                # limit audio if it exceed max_audio_length
                padded_audio = np.pad(
                    a,
                    (0, max_len - a.shape[0]),
                    mode="constant",
                    constant_values=self.padding_value,
                )
                padded_audios.append(padded_audio)

            padded_audios = np.array(padded_audios)
            # limit audio if it exceed max_audio_length
            if self.limit:
                padded_audios = padded_audios[:, : self.max_audio_length]

            return padded_audios, label

        audio_numpy = audio.numpy()
        # limit audio if it exceed max_audio_length
        if self.limit:
            audio_numpy = audio_numpy[: self.max_audio_length]
        padded_audio = np.pad(
            audio_numpy,
            (0, self.max_audio_length - audio_numpy.shape[0]),
            mode="constant",
            constant_values=self.padding_value,
        )

        audio.audio = padded_audio

        return audio, label


class AudioToSpectrogram(Transformer):
    """Read wav file with librosa and return audio and label

    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
        log_level (int): Logging level (default: logging.INFO)
    """

    def __init__(
        self,
        frame_length: int = 256,
        frame_step: int = 160,
        fft_length: int = 384,
        log_level: int = logging.INFO,
    ) -> None:
        super(AudioToSpectrogram, self).__init__(log_level=log_level)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module("librosa")
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError(
                "librosa is required to augment Audio. Please install it with `pip install librosa`."
            )

    def __call__(self, audio: Audio, label: typing.Any):
        """Compute the spectrogram of a WAV file

        Args:
            audio (Audio): Audio object
            label (Any): Label of audio

        Returns:
            np.ndarray: Spectrogram of audio
            label (Any): Label of audio
        """

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = self.librosa.stft(
            audio.numpy(),
            hop_length=self.frame_step,
            win_length=self.frame_length,
            n_fft=self.fft_length,
        ).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (
            np.std(spectrogram) + 1e-10
        )

        return spectrogram, label


class ImageShowCV2(Transformer):
    """Show image for visual inspection"""

    def __init__(
        self, verbose: bool = True, log_level: int = logging.INFO, name: str = "Image"
    ) -> None:
        """
        Args:
            verbose (bool): Whether to log label
            log_level (int): Logging level (default: logging.INFO)
            name (str): Name of window to show image
        """
        super(ImageShowCV2, self).__init__(log_level=log_level)
        self.verbose = verbose
        self.name = name
        self.thread_started = False

    def init_thread(self):
        if not self.thread_started:
            self.thread_started = True
            self.image_queue = queue.Queue()

            # Start a new thread to display the images, so that the main loop could run in multiple threads
            self.thread = threading.Thread(target=self._display_images)
            self.thread.start()

    def _display_images(self) -> None:
        """Display images in a continuous loop"""
        while True:
            image, label = self.image_queue.get()
            if isinstance(label, Image):
                cv2.imshow(self.name + "Label", label.numpy())
            cv2.imshow(self.name, image.numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def __call__(
        self, image: Image, label: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Show image for visual inspection

        Args:
            data (np.ndarray): Image data
            label (np.ndarray): Label data

        Returns:
            data (np.ndarray): Image data
            label (np.ndarray): Label data (unchanged)
        """
        # Start cv2 image display thread
        self.init_thread()

        if self.verbose:
            if isinstance(label, (str, int, float)):
                self.logger.info(f"Label: {label}")

        if isinstance(label, Detections):
            for detection in label:
                img = detection.applyToFrame(np.asarray(image.numpy()))
                image.update(img)

        # Add image to display queue
        # Sleep if queue is not empty
        while not self.image_queue.empty():
            time.sleep(0.5)

        # Add image to display queue
        self.image_queue.put((image, label))

        return image, label


# from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate


# from mltu.annotations.audio import Audio
# from mltu.annotations.detections import Detections, Detection, BboxType

""" 
Implemented image augmentors:
- RandomBrightness
- RandomRotate
- RandomErodeDilate
- RandomSharpen
- RandomGaussianBlur
- RandomSaltAndPepper
- RandomMirror
- RandomFlip
- RandomDropBlock
- RandomMosaic
- RandomZoom
- RandomColorMode
- RandomElasticTransform

Implemented audio augmentors:
- RandomAudioNoise
- RandomAudioPitchShift
- RandomAudioTimeStretch
"""


def randomness_decorator(func):
    """Decorator for randomness"""

    def wrapper(
        self, data: typing.Union[Image, Audio], annotation: typing.Any
    ) -> typing.Tuple[typing.Union[Image, Audio], typing.Any]:
        """Decorator for randomness and type checking

        Args:
            data (typing.Union[Image, Audio]): Image or Audio object to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            data (typing.Union[Image, Audio]): Adjusted image or audio
            annotation (typing.Any): Adjusted annotation
        """
        # check if image is Image object
        if not isinstance(data, (Image, Audio)):
            self.logger.error(
                f"data must be Image or Audio object, not {type(data)}, skipping augmentor"
            )
            # TODO instead of error convert image into Image object
            # TODO instead of error convert audio into Audio object
            return data, annotation

        if np.random.rand() > self._random_chance:
            return data, annotation

        # return result of function
        return func(self, data, annotation)

    return wrapper


class Augmentor:
    """Object that should be inherited by all augmentors

    Args:
        random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
        log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
    """

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
    ) -> None:
        self._random_chance = random_chance
        self._log_level = log_level
        self._augment_annotation = augment_annotation

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert (
            0 <= self._random_chance <= 1.0
        ), "random chance must be between 0.0 and 1.0"

    def augment(self, data: typing.Union[Image, Audio]):
        """Augment data"""
        raise NotImplementedError

    @randomness_decorator
    def __call__(
        self, data: typing.Union[Image, Audio], annotation: typing.Any
    ) -> typing.Tuple[typing.Union[Image, Audio], typing.Any]:
        """Randomly add noise to audio

        Args:
            data (typing.Union[Image, Audio]): Image or Audio object to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            data (typing.Union[Image, Audio]): Adjusted image or audio
            annotation (typing.Any): Adjusted annotation if necessary
        """
        data = self.augment(data)

        if self._augment_annotation and isinstance(annotation, np.ndarray):
            annotation = self.augment(annotation)

        return data, annotation


class RandomBrightness(Augmentor):
    """Randomly adjust image brightness"""

    def __init__(
        self,
        random_chance: float = 0.5,
        delta: int = 100,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
    ) -> None:
        """Randomly adjust image brightness

        Args:
            random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
            delta (int, optional): Integer value for brightness adjustment. Defaults to 100.
            log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool, optional): If True, the annotation will be adjusted as well. Defaults to False.
        """
        super(RandomBrightness, self).__init__(
            random_chance, log_level, augment_annotation
        )

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self._delta = delta

    def augment(self, image: Image, value: float) -> Image:
        """Augment image brightness"""
        hsv = np.array(image.HSV(), dtype=np.float32)

        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 2] = hsv[:, :, 2] * value

        hsv = np.uint8(np.clip(hsv, 0, 255))

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image.update(img)

        return image

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly adjust image brightness

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation if necessary
        """
        value = 1 + np.random.uniform(-self._delta, self._delta) / 255

        image = self.augment(image, value)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation, value)

        return image, annotation


class RandomRotate(Augmentor):
    """Randomly rotate image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        angle: typing.Union[int, typing.List] = 30,
        borderValue: typing.Tuple[int, int, int] = None,
        log_level: int = logging.INFO,
        augment_annotation: bool = True,
    ) -> None:
        """Randomly rotate image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            angle (int, list): Integer value or list of integer values for image rotation
            borderValue (tuple): Tuple of 3 integers, setting border color for image rotation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): If True, the annotation will be adjusted as well. Defaults to True.
        """
        super(RandomRotate, self).__init__(random_chance, log_level, augment_annotation)

        self._angle = angle
        self._borderValue = borderValue

    @staticmethod
    def rotate_image(
        image: np.ndarray,
        angle: typing.Union[float, int],
        borderValue: tuple = (0, 0, 0),
        return_rotation_matrix: bool = False,
    ) -> np.ndarray:
        # grab the dimensions of the image and then determine the centre
        height, width = image.shape[:2]
        center_x, center_y = (width // 2, height // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((height * sin) + (width * cos))
        nH = int((height * cos) + (width * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center_x
        M[1, 2] += (nH / 2) - center_y

        # perform the actual rotation and return the image
        img = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)

        if return_rotation_matrix:
            return img, M

        return img

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly rotate image

        Args:
            image (Image): Image to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Adjusted image
            annotation (typing.Any): Adjusted annotation
        """
        # check if angle is list of angles or a single angle value
        if isinstance(self._angle, list):
            angle = float(np.random.choice(self._angle))
        else:
            angle = float(np.random.uniform(-self._angle, self._angle))

        # generate random border color
        borderValue = (
            np.random.randint(0, 255, 3)
            if self._borderValue is None
            else self._borderValue
        )
        borderValue = [int(v) for v in borderValue]

        img, rotMat = self.rotate_image(
            image.numpy(), angle, borderValue, return_rotation_matrix=True
        )

        if self._augment_annotation:
            if isinstance(annotation, Image):
                # perform the actual rotation and return the annotation image
                annotation_image = self.rotate_image(
                    annotation.numpy(), angle, borderValue=(0, 0, 0)
                )
                annotation.update(annotation_image)
            elif isinstance(annotation, Detections):
                height, width = img.shape[:2]
                for detection in annotation:
                    detection.dot(rotMat, width, height)

        image.update(img)

        return image, annotation


class RandomErodeDilate(Augmentor):
    """Randomly erode and dilate image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        kernel_size: typing.Tuple[int, int] = (1, 1),
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
    ) -> None:
        """Randomly erode and dilate image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            kernel_size (tuple): Tuple of 2 integers, setting kernel size for erosion and dilation
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Boolean value to determine if annotation should be adjusted. Defaults to False.
        """
        super(RandomErodeDilate, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self._kernel_size = kernel_size
        self.kernel = np.ones(self._kernel_size, np.uint8)

    def augment(self, image: Image) -> Image:
        if np.random.rand() <= 0.5:
            img = cv2.erode(image.numpy(), self.kernel, iterations=1)
        else:
            img = cv2.dilate(image.numpy(), self.kernel, iterations=1)

        image.update(img)

        return image

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly erode and dilate image

        Args:
            image (Image): Image to be eroded and dilated
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Eroded and dilated image
            annotation (typing.Any): Adjusted annotation if necessary
        """
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

        return image, annotation


class RandomSharpen(Augmentor):
    """Randomly sharpen image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        alpha: float = 0.25,
        lightness_range: typing.Tuple = (0.75, 2.0),
        kernel: np.ndarray = None,
        kernel_anchor: np.ndarray = None,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
    ) -> None:
        """Randomly sharpen image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            alpha (float): Float between 0.0 and 1.0 setting bounds for random probability
            lightness_range (tuple): Tuple of 2 floats, setting bounds for random lightness change
            kernel (np.ndarray): Numpy array of kernel for image convolution
            kernel_anchor (np.ndarray): Numpy array of kernel anchor for image convolution
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Boolean to determine if annotation should be augmented. Defaults to False.
        """
        super(RandomSharpen, self).__init__(
            random_chance, log_level, augment_annotation
        )

        self._alpha_range = (alpha, 1.0)
        self._ligtness_range = lightness_range
        self._lightness_anchor = 8

        self._kernel = (
            np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.float32)
            if kernel is None
            else kernel
        )
        self._kernel_anchor = (
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
            if kernel_anchor is None
            else kernel_anchor
        )

        assert 0 <= alpha <= 1.0, "Alpha must be between 0.0 and 1.0"

    def augment(self, image: Image) -> Image:
        lightness = np.random.uniform(*self._ligtness_range)
        alpha = np.random.uniform(*self._alpha_range)

        kernel = (
            self._kernel_anchor * (self._lightness_anchor + lightness) + self._kernel
        )
        kernel -= self._kernel_anchor
        kernel = (1 - alpha) * self._kernel_anchor + alpha * kernel

        # Apply sharpening to each channel
        r, g, b = cv2.split(image.numpy())
        r_sharp = cv2.filter2D(r, -1, kernel)
        g_sharp = cv2.filter2D(g, -1, kernel)
        b_sharp = cv2.filter2D(b, -1, kernel)

        # Merge the sharpened channels back into the original image
        image.update(cv2.merge([r_sharp, g_sharp, b_sharp]))

        return image

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly sharpen image

        Args:
            image (Image): Image to be sharpened
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            image (Image): Sharpened image
            annotation (typing.Any): Adjusted annotation if necessary
        """
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

        return image, annotation


class RandomGaussianBlur(Augmentor):
    """Randomly erode and dilate image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        sigma: typing.Union[int, float] = 1.5,
        augment_annotation: bool = False,
    ) -> None:
        """Randomly erode and dilate image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            sigma (int, float): maximum sigma value for Gaussian blur. Defaults to 1.5.
        """
        super(RandomGaussianBlur, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self.sigma = sigma

    def augment(self, image: Image) -> Image:
        sigma = np.random.uniform(0, self.sigma)
        img = cv2.GaussianBlur(image.numpy(), (0, 0), sigma)

        image.update(img)

        return image

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly blurs an image with a Gaussian filter

        Args:
            image (Image): Image to be blurred
            annotation (typing.Any): Annotation to be blurred

        Returns:
            image (Image): Blurred image
            annotation (typing.Any): Blurred annotation if necessary
        """
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

        return image, annotation


class RandomSaltAndPepper(Augmentor):
    """Randomly add Salt and Pepper noise to image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        salt_vs_pepper: float = 0.5,
        amount: float = 0.1,
        augment_annotation: bool = False,
    ) -> None:
        """Randomly add Salt and Pepper noise to image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            salt_vs_pepper (float): ratio of salt vs pepper. Defaults to 0.5.
            amount (float): proportion of the image to be salted and peppered. Defaults to 0.1.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        """
        super(RandomSaltAndPepper, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self.salt_vs_pepper = salt_vs_pepper
        self.amount = amount

        assert 0 <= salt_vs_pepper <= 1.0, "salt_vs_pepper must be between 0.0 and 1.0"
        assert 0 <= amount <= 1.0, "amount must be between 0.0 and 1.0"

    def augment(self, image: Image) -> Image:
        img = image.numpy()
        height, width, channels = img.shape

        # Salt mode
        num_salt = int(self.amount * height * width * self.salt_vs_pepper)
        row_coords = np.random.randint(0, height, size=num_salt)
        col_coords = np.random.randint(0, width, size=num_salt)
        img[row_coords, col_coords, :] = [255, 255, channels]

        # Pepper mode
        num_pepper = int(self.amount * height * width * (1.0 - self.salt_vs_pepper))
        row_coords = np.random.randint(0, height, size=num_pepper)
        col_coords = np.random.randint(0, width, size=num_pepper)
        img[row_coords, col_coords, :] = [0, 0, channels]

        image.update(img)

        return image

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly add salt and pepper noise to an image

        Args:
            image (Image): Image to be noised
            annotation (typing.Any): Annotation to be noised

        Returns:
            image (Image): Noised image
            annotation (typing.Any): Noised annotation if necessary
        """
        image = self.augment(image)

        if self._augment_annotation and isinstance(annotation, Image):
            annotation = self.augment(annotation)

        return image, annotation


class RandomMirror(Augmentor):
    """Randomly mirror image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = True,
    ) -> None:
        """Randomly mirror image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to True.
        """
        super(RandomMirror, self).__init__(random_chance, log_level, augment_annotation)

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly mirror an image

        Args:
            image (Image): Image to be mirrored
            annotation (typing.Any): Annotation to be mirrored

        Returns:
            image (Image): Mirrored image
            annotation (typing.Any): Mirrored annotation if necessary
        """
        image = image.flip(0)
        if self._augment_annotation and isinstance(annotation, Image):
            annotation = annotation.flip(0)

        elif isinstance(annotation, Detections):
            for detection in annotation:
                detection.flip(0)

        return image, annotation


class RandomFlip(Augmentor):
    """Randomly flip image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = True,
    ) -> None:
        """Randomly mirror image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to True.
        """
        super(RandomFlip, self).__init__(random_chance, log_level, augment_annotation)

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly mirror an image

        Args:
            image (Image): Image to be flipped
            annotation (typing.Any): Annotation to be flipped

        Returns:
            image (Image): Flipped image
            annotation (typing.Any): Flipped annotation if necessary
        """
        image = image.flip(1)
        if self._augment_annotation and isinstance(annotation, Image):
            annotation = annotation.flip(1)

        elif isinstance(annotation, Detections):
            for detection in annotation:
                detection.flip(1)

        return image, annotation


class RandomDropBlock(Augmentor):
    """Randomly drop block from image"""

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        block_size_percentage: float = 0.05,
        keep_prob: float = 0.7,
    ) -> None:
        """Randomly drop block from image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
            block_size_percentage (float): drop block size percentage relative to image size. Defaults to 0.05.
            keep_prob (float): Probability of keeping the block. Defaults to 0.7.
        """
        super(RandomDropBlock, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self.block_size_percentage = block_size_percentage
        self.keep_prob = keep_prob

    @staticmethod
    def dropblock(image, block_percent=0.05, keep_prob=0.7):
        height, width = image.shape[:2]
        block_size = int(min(height, width) * block_percent)
        mask = np.ones((height, width), dtype=bool)

        for i in range(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                if np.random.rand() > keep_prob:
                    mask[i : i + block_size, j : j + block_size] = False

        dropped_image = image * mask[..., np.newaxis]
        return dropped_image

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly drop block from image

        Args:
            image (Image): Image to be dropped
            annotation (typing.Any): Annotation to be dropped

        Returns:
            image (Image): Dropped image
            annotation (typing.Any): Dropped annotation if necessary
        """
        img = self.dropblock(image.numpy(), self.block_size_percentage, self.keep_prob)
        image.update(img)

        return image, annotation


class RandomMosaic(Augmentor):
    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = True,
        target_size: typing.Tuple[int, int] = None,
    ) -> None:
        """Randomly merge 4 images into one mosaic image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
            target_size (tuple): Tuple of 2 integers, setting target size for mosaic image. Defaults to None.
        """
        super(RandomMosaic, self).__init__(random_chance, log_level, augment_annotation)
        self.target_size = target_size
        self.images = []
        self.annotations = []

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """R

        Args:
            image (Image): Image to be used for mosaic
            annotation (typing.Any): Annotation to be used for mosaic

        Returns:
            image (Image): Mosaic image
            annotation (typing.Any): Mosaic annotation if necessary
        """
        if not isinstance(annotation, Detections):
            self.logger.error(
                f"annotation must be Detections object, not {type(annotation)}, skipping augmentor"
            )
            return image, annotation

        self.images.append(image.numpy())
        self.annotations.append(annotation)

        if len(self.images) >= 4:
            # merge images and annotations into one image and annotation
            if self.target_size is None:
                # pick smalles target size and resize all images to that size
                target_size = (
                    min([img.shape[0] for img in self.images]),
                    min([img.shape[1] for img in self.images]),
                )
            else:
                target_size = self.target_size

            images = [cv2.resize(img, target_size) for img in self.images[:4]]
            detections = []
            new_img = np.concatenate(
                [
                    np.concatenate(images[:2], axis=1),
                    np.concatenate(images[2:4], axis=1),
                ],
                axis=0,
            )

            height, width = new_img.shape[:2]
            for index, annotation in enumerate(self.annotations[:4]):
                if isinstance(annotation, Detections):
                    for detection in annotation:
                        xywh = np.array(detection.xywh) / 2

                        if index in [1, 3]:
                            xywh[0] = xywh[0] + 0.5

                        if index in [2, 3]:
                            xywh[1] = xywh[1] + 0.5

                        new_detection = Detection(
                            xywh,
                            label=detection.label,
                            labels=detection.labels,
                            confidence=detection.confidence,
                            image_path=detection.image_path,
                            width=width,
                            height=height,
                            relative=True,
                        )
                        detections.append(new_detection)

            new_detections = Detections(
                labels=annotation.labels,
                width=width,
                height=height,
                detections=detections,
            )

            image.update(new_img)

            self.images = self.images[4:]
            self.annotations = self.annotations[4:]

            return image, new_detections

        return image, annotation


class RandomZoom(Augmentor):
    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = True,
        object_crop_percentage: float = 0.5,
    ) -> None:
        """Randomly zoom into an image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
            object_crop_percentage (float): Percentage of the object allowed to be cropped. Defaults to 0.5.
        """
        super(RandomZoom, self).__init__(random_chance, log_level, augment_annotation)
        self.object_crop_percentage = object_crop_percentage

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly zoom an image

        Args:
            image (Image): Image to be used for zoom
            annotation (typing.Any): Annotation to be used for zoom

        Returns:
            image (Image): Zoomed image
            annotation (typing.Any): Zoomed annotation if necessary
        """
        if isinstance(annotation, Detections) and self._augment_annotation:

            dets = np.array([detection.xyxy for detection in annotation])
            min_left = np.min(dets[:, 0])
            min_top = np.min(dets[:, 1])
            max_right = np.max(dets[:, 2])
            max_bottom = np.max(dets[:, 3])

            # Calculate the size of the object
            object_width = max_right - min_left
            object_height = max_bottom - min_top

            crop_xmin = np.random.uniform(
                0, min_left + 0.25 * object_width * self.object_crop_percentage
            )
            crop_ymin = np.random.uniform(
                0, min_top + 0.25 * object_height * self.object_crop_percentage
            )
            crop_xmax = np.random.uniform(
                max_right - 0.25 * object_width * self.object_crop_percentage, 1
            )
            crop_ymax = np.random.uniform(
                max_bottom - 0.25 * object_height * self.object_crop_percentage, 1
            )

            crop_min_max = np.array([crop_xmin, crop_ymin, crop_xmax, crop_ymax])
            new_xyxy = (
                crop_min_max
                * np.array([image.width, image.height, image.width, image.height])
            ).astype(int)
            new_image = image.numpy()[
                new_xyxy[1] : new_xyxy[3], new_xyxy[0] : new_xyxy[2]
            ]
            image.update(new_image)

            crop_min_ratio = np.array([crop_xmin, crop_ymin, crop_xmin, crop_ymin])
            crop_max_ratio = np.array([crop_xmax, crop_ymax, crop_xmax, crop_ymax])
            new_dets = (dets - crop_min_ratio) / (crop_max_ratio - crop_min_ratio)

            detections = []
            for detection, new_det in zip(annotation, new_dets):
                new_detection = Detection(
                    new_det,
                    label=detection.label,
                    labels=detection.labels,
                    confidence=detection.confidence,
                    image_path=detection.image_path,
                    width=image.width,
                    height=image.height,
                    relative=True,
                    bbox_type=BboxType.XYXY,
                )

                detections.append(new_detection)

            annotation = Detections(
                labels=annotation.labels,
                width=image.width,
                height=image.height,
                detections=detections,
            )

        return image, annotation


class RandomColorMode(Augmentor):
    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
    ) -> None:
        """Randomly change color mode of an image

        Args:
            random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
            log_level (int): Log level for the augmentor. Defaults to logging.INFO.
            augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        """
        super(RandomColorMode, self).__init__(
            random_chance, log_level, augment_annotation
        )

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly change color mode of an image

        Args:
            image (Image): Image to be used for color mode change
            annotation (typing.Any): Annotation to be used for color mode change

        Returns:
            image (Image): Color mode changed image
            annotation (typing.Any): Color mode changed annotation if necessary
        """
        color_mode = np.random.choice(
            [
                cv2.COLOR_BGR2GRAY,
                cv2.COLOR_BGR2HSV,
                cv2.COLOR_BGR2LAB,
                cv2.COLOR_BGR2YCrCb,
                cv2.COLOR_BGR2RGB,
            ]
        )
        new_image = cv2.cvtColor(image.numpy(), color_mode)
        if color_mode == cv2.COLOR_BGR2GRAY:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
        image.update(new_image)

        return image, annotation


class RandomElasticTransform(Augmentor):
    """Randomly apply elastic transform to an image

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        alpha_range (tuple): Tuple of 2 floats, setting bounds for random alpha value. Defaults to (0, 0.1).
        sigma_range (tuple): Tuple of 2 floats, setting bounds for random sigma value. Defaults to (0.01, 0.02).
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
    """

    def __init__(
        self,
        random_chance: float = 0.5,
        alpha_range: tuple = (0, 0.1),
        sigma_range: tuple = (0.01, 0.02),
        log_level: int = logging.INFO,
        augment_annotation: bool = True,
    ) -> None:
        super(RandomElasticTransform, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    @staticmethod
    def elastic_transform(
        image: np.ndarray, alpha: float, sigma: float
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply elastic transform to an image

        Args:
            image (np.ndarray): Image to be used for elastic transform
            alpha (float): Alpha value for elastic transform
            sigma (float): Sigma value for elastic transform

        Returns:
            remap_fn (np.ndarray): Elastic transformed image
            dx (np.ndarray): X-axis displacement
            dy (np.ndarray): Y-axis displacement
        """
        height, width, channels = image.shape
        dx = np.random.rand(height, width).astype(np.float32) * 2 - 1
        dy = np.random.rand(height, width).astype(np.float32) * 2 - 1

        cv2.GaussianBlur(dx, (0, 0), sigma, dst=dx)
        cv2.GaussianBlur(dy, (0, 0), sigma, dst=dy)

        dx *= alpha
        dy *= alpha

        x, y = np.meshgrid(np.arange(width), np.arange(height))

        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)

        remap_fn = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return remap_fn, dx, dy

    @randomness_decorator
    def __call__(
        self, image: Image, annotation: typing.Any
    ) -> typing.Tuple[Image, typing.Any]:
        """Randomly apply elastic transform to an image

        Args:
            image (Image): Image to be used for elastic transform
            annotation (typing.Any): Annotation to be used for elastic transform

        Returns:
            image (Image): Elastic transformed image
            annotation (typing.Any): Elastic transformed annotation if necessary
        """
        alpha = image.width * np.random.uniform(*self.alpha_range)
        sigma = image.width * np.random.uniform(*self.sigma_range)
        new_image, dx, dy = self.elastic_transform(image.numpy(), alpha, sigma)
        image.update(new_image)

        if isinstance(annotation, Detections) and self._augment_annotation:
            detections = []
            for detection in annotation:
                x_min, y_min, x_max, y_max = detection.xyxy_abs
                x_max = min(x_max, dx.shape[1] - 1)
                y_max = min(y_max, dy.shape[0] - 1)
                new_x_min = min(max(0, x_min + dx[y_min, x_min]), image.width - 1)
                new_y_min = min(max(0, y_min + dy[y_min, x_min]), image.height - 1)
                new_x_max = min(max(0, x_max + dx[y_max, x_max]), image.width - 1)
                new_y_max = min(max(0, y_max + dy[y_max, x_max]), image.height - 1)
                detections.append(
                    Detection(
                        [new_x_min, new_y_min, new_x_max, new_y_max],
                        label=detection.label,
                        labels=detection.labels,
                        confidence=detection.confidence,
                        image_path=detection.image_path,
                        width=image.width,
                        height=image.height,
                        relative=False,
                        bbox_type=BboxType.XYXY,
                    )
                )

            annotation = Detections(
                labels=annotation.labels,
                width=image.width,
                height=image.height,
                detections=detections,
            )

        return image, annotation


class RandomAudioNoise(Augmentor):
    """Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        max_noise_ratio (float): Maximum noise ratio to be added to audio. Defaults to 0.1.
    """

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        max_noise_ratio: float = 0.1,
    ) -> None:
        super(RandomAudioNoise, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self.max_noise_ratio = max_noise_ratio

    def augment(self, audio: Audio) -> Audio:
        noise = np.random.uniform(-1, 1, len(audio))
        noise_ratio = np.random.uniform(0, self.max_noise_ratio)
        audio_noisy = audio + noise_ratio * noise

        return audio_noisy


class RandomAudioPitchShift(Augmentor):
    """Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        max_n_steps (int): Maximum number of steps to shift audio. Defaults to 5.
    """

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        max_n_steps: int = 5,
    ) -> None:
        super(RandomAudioPitchShift, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self.max_n_steps = max_n_steps

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module("librosa")
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError(
                "librosa is required to augment Audio. Please install it with `pip install librosa`."
            )

    def augment(self, audio: Audio) -> Audio:
        random_n_steps = np.random.randint(-self.max_n_steps, self.max_n_steps)
        # changing default res_type "kaiser_best" to "linear" for speed and memory efficiency
        shift_audio = self.librosa.effects.pitch_shift(
            audio.numpy(),
            sr=audio.sample_rate,
            n_steps=random_n_steps,
            res_type="linear",
        )
        audio.audio = shift_audio

        return audio


class RandomAudioTimeStretch(Augmentor):
    """Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        min_rate (float): Minimum rate to stretch audio. Defaults to 0.8.
        max_rate (float): Maximum rate to stretch audio. Defaults to 1.2.
    """

    def __init__(
        self,
        random_chance: float = 0.5,
        log_level: int = logging.INFO,
        augment_annotation: bool = False,
        min_rate: float = 0.8,
        max_rate: float = 1.2,
    ) -> None:
        super(RandomAudioTimeStretch, self).__init__(
            random_chance, log_level, augment_annotation
        )
        self.min_rate = min_rate
        self.max_rate = max_rate

        try:
            librosa.__version__
        except ImportError:
            raise ImportError(
                "librosa is required to augment Audio. Please install it with `pip install librosa`."
            )

    def augment(self, audio: Audio) -> Audio:
        random_rate = np.random.uniform(self.min_rate, self.max_rate)
        stretch_audio = librosa.effects.time_stretch(audio.numpy(), rate=random_rate)
        audio.audio = stretch_audio

        return audio


# from mltu.annotations.images import CVImage
class Image(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def center(self) -> tuple:
        pass

    @abstractmethod
    def RGB(self) -> np.ndarray:
        pass

    @abstractmethod
    def HSV(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, image: np.ndarray):
        pass

    @abstractmethod
    def flip(self, axis: int = 0):
        pass

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass


class CVImage(Image):
    """Image class for storing image data and metadata (opencv based)

    Args:
        image (str or np.ndarray): Path to image or numpy.ndarray
        method (int, optional): OpenCV method for reading image. Defaults to cv2.IMREAD_COLOR.
        path (str, optional): Path to image. Defaults to "".
        color (str, optional): Color format of image. Defaults to "BGR".
    """

    init_successful = False

    def __init__(
        self,
        image: typing.Union[str, np.ndarray],
        method: int = cv2.IMREAD_COLOR,
        path: str = "",
        color: str = "BGR",
    ) -> None:
        super().__init__()

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self._image = cv2.imread(image, method)
            self.path = image
            self.color = "BGR"

        elif isinstance(image, np.ndarray):
            self._image = image
            self.path = path
            self.color = color

        else:
            raise TypeError(
                f"Image must be either path to image or numpy.ndarray, not {type(image)}"
            )

        self.method = method

        if self._image is None:
            return None

        self.init_successful = True

        # save width, height and channels
        self.width = self._image.shape[1]
        self.height = self._image.shape[0]
        self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, value: np.ndarray):
        self._image = value

    @property
    def shape(self) -> tuple:
        return self._image.shape

    @property
    def center(self) -> tuple:
        return self.width // 2, self.height // 2

    def RGB(self) -> np.ndarray:
        if self.color == "RGB":
            return self._image
        elif self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def HSV(self) -> np.ndarray:
        if self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        elif self.color == "RGB":
            return cv2.cvtColor(self._image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def update(self, image: np.ndarray):
        if isinstance(image, np.ndarray):
            self._image = image

            # save width, height and channels
            self.width = self._image.shape[1]
            self.height = self._image.shape[0]
            self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

            return self

        else:
            raise TypeError(f"image must be numpy.ndarray, not {type(image)}")

    def flip(self, axis: int = 0):
        """Flip image along x or y axis

        Args:
            axis (int, optional): Axis along which image will be flipped. Defaults to 0.

        Returns:
            Object with flipped points
        """
        # axis must be either 0 or 1
        if axis not in [0, 1]:
            raise ValueError(f"axis must be either 0 or 1, not {axis}")

        self._image = self._image[:, ::-1] if axis == 0 else self._image[::-1]

        return self

    def numpy(self) -> np.ndarray:
        return self._image

    def __call__(self) -> np.ndarray:
        return self._image


class PillowImage(Image):
    """Image class for storing image data and metadata (pillow based)

    Args:
        image (str): Path to image
    """

    init_successful = False

    def __init__(self, image: str) -> None:
        super().__init__()

        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self.path = image
            self._image = PilImage.open(image)

            self.init_successful = True
        else:
            raise TypeError("Image must be a path to an image")

        if self.is_animated:
            # initialize whatever attributes we can already determine at this stage, i.e. width & height.
            self.width = self._image.width
            self.height = self._image.height
            self.channels = None
        else:
            self._init_attributes()

    @property
    def is_animated(self) -> bool:
        return hasattr(self._image, "is_animated") and self._image.is_animated

    @property
    def image(self) -> np.ndarray:
        if self.is_animated:
            raise Exception("convert to single image first")

        return np.asarray(self._image)

    @image.setter
    def image(self, value: np.ndarray):
        self._image = PilImage.fromarray(value)

    @property
    def shape(self) -> tuple:
        return self.image.shape

    @property
    def center(self) -> tuple:
        return self.width // 2, self.height // 2

    def RGB(self) -> np.ndarray:
        if self.color == "RGB":
            return self.image
        elif self.color == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def HSV(self) -> np.ndarray:
        if self.color == "BGR":
            return cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        elif self.color == "RGB":
            return cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def _init_attributes(self):
        self.color = self._image.mode

        # save width, height and channels
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.channels = 1 if len(self.image.shape) == 2 else self.image.shape[2]

    def update(self, image: PilImage.Image):
        if isinstance(image, PilImage.Image):
            self._image = image
        elif isinstance(image, np.ndarray):
            self._image = PilImage.fromarray(image)
        else:
            raise TypeError(
                f"image must be a Pillow Image or np.ndarray, not {type(image)}"
            )

        if not self.is_animated:
            self._init_attributes()

        return self

    def flip(self, axis: int = 0):
        """Flip image along x or y axis

        Args:
            axis (int, optional): Axis along which image will be flipped. Defaults to 0.

        Returns:
            Object with flipped points
        """
        # axis must be either 0 or 1
        if axis not in [0, 1]:
            raise ValueError(f"axis must be either 0 or 1, not {axis}")

        if self.is_animated:
            raise Exception("convert to single image first")

        if axis == 0:
            self._image = PilImage.fromarray(np.asarray(self._image)[:, ::-1])
        else:
            self._image = PilImage.fromarray(np.asarray(self._image)[::-1])

        return self

    def numpy(self) -> np.ndarray:
        return self.image

    def __call__(self) -> np.ndarray:
        return self.image

    def pillow(self) -> PilImage.Image:
        return self._image


# from model import train_model
# from mltu.tensorflow.model_utils import residual_block
class CustomModel(Model):
    """Custom TensorFlow model for debugging training process purposes"""

    def train_step(self, train_data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs, targets = train_data
        with tf.GradientTape() as tape:
            results = self(inputs, training=True)
            loss = self.compiled_loss(
                targets, results, regularization_losses=self.losses
            )
            gradients = tape.gradient(loss, self.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(targets, results)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, test_data):
        inputs, targets = test_data
        # Get prediction from model
        results = self(inputs, training=False)

        # Update the loss
        self.compiled_loss(targets, results, regularization_losses=self.losses)

        # Update the metrics
        self.compiled_metrics.update_state(targets, results)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


def activation_layer(layer, activation: str = "relu", alpha: float = 0.1) -> tf.Tensor:
    """Activation layer wrapper for LeakyReLU and ReLU activation functions
    Args:
        layer: tf.Tensor
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)
    Returns:
        tf.Tensor
    """
    if activation == "relu":
        layer = layers.ReLU()(layer)
    elif activation == "leaky_relu":
        layer = layers.LeakyReLU(alpha=alpha)(layer)

    return layer


def residual_block(
    x: tf.Tensor,
    filter_num: int,
    strides: typing.Union[int, list] = 2,
    kernel_size: typing.Union[int, list] = 3,
    skip_conv: bool = True,
    padding: str = "same",
    kernel_initializer: str = "he_uniform",
    activation: str = "relu",
    dropout: float = 0.2,
):
    # Create skip connection tensor
    x_skip = x

    # Perform 1-st convolution
    x = layers.Conv2D(
        filter_num,
        kernel_size,
        padding=padding,
        strides=strides,
        kernel_initializer=kernel_initializer,
    )(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)

    # Perform 2-nd convoluti
    x = layers.Conv2D(
        filter_num, kernel_size, padding=padding, kernel_initializer=kernel_initializer
    )(x)
    x = layers.BatchNormalization()(x)

    # Perform 3-rd convolution if skip_conv is True, matchin the number of filters and the shape of the skip connection tensor
    if skip_conv:
        x_skip = layers.Conv2D(
            filter_num,
            1,
            padding=padding,
            strides=strides,
            kernel_initializer=kernel_initializer,
        )(x_skip)

    # Add x and skip connection and apply activation function
    x = layers.Add()([x, x_skip])
    x = activation_layer(x, activation=activation)

    # Apply dropout
    if dropout:
        x = layers.Dropout(dropout)(x)

    return x


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):

    inputs = layers.Input(shape=input_dim, name="input")

    # normalize images here instead in preprocessing step
    input = layers.Lambda(lambda x: x / 255)(inputs)

    x1 = residual_block(
        input, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout
    )

    x2 = residual_block(
        x1, 16, activation=activation, skip_conv=True, strides=2, dropout=dropout
    )
    x3 = residual_block(
        x2, 16, activation=activation, skip_conv=False, strides=1, dropout=dropout
    )

    x4 = residual_block(
        x3, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout
    )
    x5 = residual_block(
        x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout
    )

    x6 = residual_block(
        x5, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout
    )
    x7 = residual_block(
        x6, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout
    )

    x8 = residual_block(
        x7, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout
    )
    x9 = residual_block(
        x8, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout
    )

    squeezed = layers.Reshape((x9.shape[-3] * x9.shape[-2], x9.shape[-1]))(x9)

    blstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(squeezed)
    blstm = layers.Dropout(dropout)(blstm)

    output = layers.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=inputs, outputs=output)
    return model


# from configs import ModelConfigs
# from mltu.configs import BaseModelConfigs
class BaseModelConfigs:
    def __init__(self):
        self.model_path = None

    def serialize(self):
        class_attributes = {
            key: value
            for (key, value) in type(self).__dict__.items()
            if key not in ["__module__", "__init__", "__doc__", "__annotations__"]
        }
        instance_attributes = self.__dict__

        # first init with class attributes then apply instance attributes overwriting any existing duplicate attributes
        all_attributes = class_attributes.copy()
        all_attributes.update(instance_attributes)

        return all_attributes

    def save(self, name: str = "configs.yaml"):
        if self.model_path is None:
            raise Exception("Model path is not specified")

        # create directory if not exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, name), "w") as f:
            yaml.dump(self.serialize(), f)

    @staticmethod
    def load(configs_path: str):
        with open(configs_path, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        config = BaseModelConfigs()
        for key, value in configs.items():
            setattr(config, key, value)

        return config


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(
            "Models/02_captcha_to_text", datetime.strftime(datetime.now(), "%Y%m%d%H%M")
        )
        self.vocab = ""
        self.height = 50
        self.width = 200
        self.max_text_length = 0
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.train_epochs = 1000
        self.train_workers = 20


def download_and_unzip(url, extract_to="Datasets"):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)


# Backup Training Set
if not os.path.exists(os.path.join("Datasets", "captcha_images_v2")):
    download_and_unzip(
        "https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip",
        extract_to="Datasets",
    )

# Create a list of all the images and labels in the dataset
dataset, vocab, max_len = [], set(), 0
captcha_path = os.path.join("Datasets", "captcha_images_v2")
for file in os.listdir(captcha_path):
    file_path = os.path.join(captcha_path, file)
    label = os.path.splitext(file)[0]  # Get the file name without the extension
    dataset.append([file_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len
configs.save()

# Create a data provider for the dataset
data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configs.width, configs.height),
        LabelIndexer(configs.vocab),
        LabelPadding(
            max_word_length=configs.max_text_length, padding_value=len(configs.vocab)
        ),
    ],
)
# Split the dataset into training and validation sets
train_data_provider, val_data_provider = data_provider.split(split=0.9)

# Augment training data with random brightness, rotation and erode/dilate
train_data_provider.augmentors = [
    RandomBrightness(),
    RandomRotate(),
    RandomErodeDilate(),
]

# Creating TensorFlow model architecture
model = train_model(
    input_dim=(configs.height, configs.width, 3),
    output_dim=len(configs.vocab),
)

# Compile the model and print summary
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
    run_eagerly=False,
)
model.summary(line_length=110)
# Define path to save the model
os.makedirs(configs.model_path, exist_ok=True)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1)
checkpoint = ModelCheckpoint(
    f"{configs.model_path}/model.h5",
    monitor="val_CER",
    verbose=1,
    save_best_only=True,
    mode="min",
)
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(
    monitor="val_CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode="auto"
)
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[
        earlystopper,
        checkpoint,
        trainLogger,
        reduceLROnPlat,
        tb_callback,
        model2onnx,
    ],
    workers=configs.train_workers,
)

# Save training and validation datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
