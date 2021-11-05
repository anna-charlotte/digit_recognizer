import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Callable

import matplotlib.pyplot as plt
import metriculous
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2


@dataclass
class DataSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def __add__(self, other):
        added_data_splits = DataSplits(
            x_train=np.append(self.x_train, other.x_train),
            y_train=np.append(self.y_train, other.y_train),
            x_val=np.append(self.x_val, other.x_val),
            y_val=np.append(self.y_val, other.y_val),
            x_test=np.append(self.x_test, other.x_test),
            y_test=np.append(self.y_test, other.y_test),
        )
        return added_data_splits

    def get_shape(self):
        return (
            len(self.x_train),
            len(self.y_train),
            len(self.x_val),
            len(self.y_val),
            len(self.x_test),
            len(self.y_test),
        )


def stack_to_rgb_image(grey_scale_image: np.ndarray, axis=-1) -> np.ndarray:
    assert grey_scale_image.ndim == 2
    output = np.stack((grey_scale_image, grey_scale_image, grey_scale_image), axis=axis)
    return output


def raw_df_to_x_y(
        raw_data_frame: pd.DataFrame,
        img_height=28,
        img_width=28,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a tuple (x, y_one_hot) if data frame has a 'label' column, otherwise (x, None).
    """
    num_images = len(raw_data_frame)
    if "label" in raw_data_frame.columns:
        df_x = raw_data_frame.drop(labels="label", axis=1)
    else:
        df_x = raw_data_frame
    num_pixels = df_x.shape[1]
    assert num_pixels == img_height * img_width

    x = np.array(
        [
            stack_to_rgb_image(
                grey_scale_image=np.reshape(pixels, (img_height, img_width))
            )
            for pixels in df_x.values
        ]
    )
    y = raw_data_frame["label"].values if "label" in raw_data_frame.columns else None

    assert y is None or (y.shape == (num_images,))
    print(f"x.shape = {x.shape}")

    return x, y


def one_hot_encoding(input_array: np.ndarray, num_classes: int):
    assert input_array.ndim == 1
    return np.eye(num_classes)[input_array]


def make_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    model = keras.models.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(rate=0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def split_into_train_val_test_set(
        x, y, validation_percentage=0.1, test_percentage=0.02
):
    p = np.random.permutation(len(x))
    x_shuffled = x[p]
    y_shuffled = y[p]
    split_for_val = math.floor(len(x) * validation_percentage)
    split_for_test = math.floor(len(x) * test_percentage + split_for_val)

    datasplit = DataSplits(
        x_val=x_shuffled[:split_for_val],
        y_val=y_shuffled[:split_for_val],
        x_test=x_shuffled[split_for_val:split_for_test],
        y_test=y_shuffled[split_for_val:split_for_test],
        x_train=x_shuffled[split_for_test:],
        y_train=y_shuffled[split_for_test:],
    )
    return datasplit


def read_in_images_as_dataframe(directory: Path, img_height, img_width):
    columns_for_df_max = img_height * img_width
    column_names = ["label"]
    for i in range(columns_for_df_max):
        col = f"pixel{i}"
        column_names.append(col)

    data = []

    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            class_label = re.search("class(.)_", str(file)).group(1)

            img = Image.open(Path(directory / file))

            img_array = np.array(img)
            img_vector = img_array.flatten()
            labeled_img_vector = np.insert(img_vector, 0, class_label)
            data.append(labeled_img_vector)

    dataframe = pd.DataFrame(data=data, columns=column_names)

    return dataframe


def multiply_pixels(
        image: np.ndarray, vertical=2, horizontal=2, img_height=28, img_width=28
):
    new_image = np.zeros(shape=(img_height * vertical, img_width * horizontal))

    for i in range(img_height):
        for j in range(img_width):
            pixel = image[i][j]
            for m in range(horizontal):
                for n in range(vertical):
                    x = n + (i * vertical)
                    y = m + (j * horizontal)
                    new_image[x][y] = pixel
    return new_image


def display_image(image: np.ndarray, title=""):
    assert image.ndim == 3
    assert image.shape[2] == 3
    plt.imshow(image, interpolation="nearest")
    plt.title(title)
    plt.show()


def display_wrong_predictions(
        x_true: np.ndarray, y_true: np.ndarray, predictions: np.ndarray, save=False
):
    print("Display wrong predictions...")
    count = 0
    for i in range(len(predictions)):
        class_pred = np.argmax(predictions[i])
        class_true = np.argmax(y_true[i])
        if class_pred != class_true:
            plt.imshow(x_true[i], interpolation="nearest")
            plt.title(str(class_pred) + " != " + str(class_true) + " (pred != gt)")
            plt.show()
            count += 1


def save_predictions_for_submission(
        df_submission, model, path_to_dir: Path, title="prediction"
):
    x_submission, _ = raw_df_to_x_y(raw_data_frame=df_submission)

    print("Predict...")
    predictions = model.predict(x_submission)

    print("Save predictions...")
    file_path = path_to_dir / title
    with open(file_path, "w") as file:
        file.write("ImageId,Label\n")
        count = 1
        for row in predictions:
            file.write(
                str(count) + "," + str(max(range(len(row)), key=row.__getitem__)) + "\n"
            )
            count += 1


def save_this_script(destination_dir: Path):
    shutil.copy(__file__, destination_dir)


def save_model_comparison_in_cwd(x_test, y_test, model, path: Path, class_names):
    print("Evaluate model...")
    model_prediction = model.predict(x_test)
    ground_truth = one_hot_encoding(y_test, 10)

    metriculous.compare_classifiers(
        ground_truth=ground_truth,
        model_predictions=[model_prediction],
        class_names=class_names,
        filter_figures=lambda n: "scatter" not in n.lower(),
    ).save_html(path / "comparison.html").display()


def add_some_of_array_to_all_of_array(
        a: np.ndarray, b: np.ndarray, fraction_from_b_in_result: float, seed: int = 42
) -> np.ndarray:
    assert 0.0 <= fraction_from_b_in_result <= 1.0, fraction_from_b_in_result
    num_of_b_to_use = int(
        len(a) * fraction_from_b_in_result / (1.0 - fraction_from_b_in_result)
    )
    rng = np.random.RandomState(seed)
    indices_of_b_to_use = rng.choice(
        list(range(len(b))),
        size=num_of_b_to_use,
        replace=False,
    )
    subset_of_b = b[indices_of_b_to_use]
    assert subset_of_b.shape[1:] == b.shape[1:]
    assert subset_of_b.shape[0] == num_of_b_to_use
    concatenated_a_subset_b = np.concatenate((a, subset_of_b), axis=0)
    p = rng.permutation(len(concatenated_a_subset_b))

    return concatenated_a_subset_b[p]


def add_some_of_the_generated_data_to_val_and_train(
        all_of: DataSplits,
        with_some_of: DataSplits,
        fraction_of_generated_data_to_use: float,
        seed: int = 42,
) -> DataSplits:
    new_x_train = add_some_of_array_to_all_of_array(all_of.x_train,
                                                    with_some_of.x_train,
                                                    fraction_of_generated_data_to_use,
                                                    seed)
    new_y_train = add_some_of_array_to_all_of_array(all_of.y_train,
                                                    with_some_of.y_train,
                                                    fraction_of_generated_data_to_use,
                                                    seed)

    new_x_val = add_some_of_array_to_all_of_array(all_of.x_val, with_some_of.x_val,
                                                  fraction_of_generated_data_to_use,
                                                  seed)
    new_y_val = add_some_of_array_to_all_of_array(all_of.y_val, with_some_of.y_val,
                                                  fraction_of_generated_data_to_use,
                                                  seed)

    return DataSplits(x_train=new_x_train,
                      y_train=new_y_train,
                      x_val=new_x_val,
                      y_val=new_y_val,
                      x_test=all_of.x_test,
                      y_test=all_of.y_test
                      )


def trial_run(
        make_model: Callable[[], Sequential],
        optimizer: OptimizerV2,
        loss: Loss,
        data_splits: DataSplits,
        data_splits_generated_data: DataSplits,
        max_epochs: int,
        experiment_dir: Path,
        num_classes: int = 10,
        metrics: str = "accuracy",
        fraction_of_generated_data_to_use: float = 0.0,
        display_wrong_pred: bool = False,
        save_predictions: bool = True,
        save_evaluation: bool = True,
        seed: int = 42,
):
    trial_dir = experiment_dir / f'trial_{time.strftime("%Y-%m-%d-%H%M%S")}'
    Path.mkdir(trial_dir)

    data_splits_final = add_some_of_the_generated_data_to_val_and_train(
        all_of=data_splits,
        with_some_of=data_splits_generated_data,
        fraction_of_generated_data_to_use=fraction_of_generated_data_to_use,
        seed=seed)

    y_train_one_hot = one_hot_encoding(
        data_splits_final.y_train, num_classes=num_classes
    )
    y_val_one_hot = one_hot_encoding(data_splits_final.y_val, num_classes=num_classes)

    model = make_model()
    print("Compiling model...")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print("Model summary:\n")
    model.summary()
    print("Fitting model...")
    model.fit(
        x=data_splits_final.x_train,
        y=y_train_one_hot,
        epochs=max_epochs,
        validation_data=(data_splits_final.x_val, y_val_one_hot),
        shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3)],
    )

    model.save(filepath=trial_dir / "model.h5", save_format="h5")

    # display wrong predictions from validation set
    if display_wrong_pred:
        display_wrong_predictions(
            x_true=data_splits_final.x_val,
            y_true=y_val_one_hot,
            predictions=model.predict(data_splits_final.x_val),
        )

    # predict
    if save_predictions:
        print("Loading test CSV file...")
        df_submission = pd.read_csv(Path(__file__).parent.parent / "data/test.csv")
        save_predictions_for_submission(
            df_submission=df_submission,
            model=model,
            path_to_dir=trial_dir,
        )

    # evaluate
    if save_evaluation:
        save_model_comparison_in_cwd(
            x_test=data_splits_final.x_test,
            y_test=data_splits_final.y_test,
            model=model,
            path=trial_dir,
            class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        )


def main(mnist_height: int = 28, mnist_width: int = 28, mnist_num_classes: int = 10,
         use_sample_data: bool = False
         ):
    experiment_dir = (
            Path(__file__).parent.parent
            / "experiments"
            / f'experiment_{time.strftime("%Y-%m-%d-%H%M%S")}'
    )

    Path.mkdir(experiment_dir)
    os.chdir(experiment_dir)

    save_this_script(experiment_dir)

    print("Loading training CSV file...")
    if use_sample_data:
        df_train = pd.read_csv(Path(__file__).parent.parent / "data/train_sample.csv")
    else:
        df_train = pd.read_csv(Path(__file__).parent.parent / "data/train.csv")

    print("Loading generated images...")
    if use_sample_data:
        df_train_generated_data = read_in_images_as_dataframe(
            (Path(__file__).parent.parent / "data_generated/test_data_generated/"),
            img_height=mnist_height,
            img_width=mnist_width,
        )
    else:
        df_train_generated_data = read_in_images_as_dataframe(
            (Path(
                __file__).parent.parent / "data_generated/data_generated_2021-10-20-120446/"),
            img_height=mnist_height,
            img_width=mnist_width,
        )

    print("Converting data frames to arrays...")
    x_train, y_train = raw_df_to_x_y(
        raw_data_frame=df_train
    )
    x_train_generated_data, y_train_generated_data = raw_df_to_x_y(
        raw_data_frame=df_train_generated_data
    )

    for i in range(10):
        display_image(x_train[i], title=f"{i}_img.jpg")
        print("display")
        # display_image(x_train[i], title=f"Index: {i}   Label: {y_train[i]}")

    print("Split labeled data into training, validation and test set...")
    data_splits = split_into_train_val_test_set(
        x_train, y_train, validation_percentage=0.1, test_percentage=0.01
    )
    data_splits_generated_data = split_into_train_val_test_set(
        x_train_generated_data,
        y_train_generated_data,
        validation_percentage=0.1,
        test_percentage=0.0,
    )

    # https://keras.io/api/applications/#usage-examples-for-image-classification-models

    def make_simple_model() -> Sequential:
        return make_simple_cnn(
            input_shape=(mnist_height, mnist_width, 3), num_classes=mnist_num_classes
        )

    def make_resnet():
        return ResNet50(
            input_shape=(mnist_height, mnist_width, 3),
            include_top=True,
            weights=None,
            classes=10,
        )

    for learning_rate in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, ):
        for model in [make_simple_model]:
            for data_to_use in [0.3, 0.4, 0.5]:
                # for model in (make_simple_model, make_resnet):
                trial_run(
                    make_model=model,
                    data_splits=data_splits,
                    data_splits_generated_data=data_splits_generated_data,
                    fraction_of_generated_data_to_use=data_to_use,
                    optimizer=optimizers.Adam(learning_rate=learning_rate),
                    max_epochs=55,
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    save_evaluation=True,
                    save_predictions=False,
                    experiment_dir=experiment_dir,
                )


if __name__ == "__main__":
    main()
