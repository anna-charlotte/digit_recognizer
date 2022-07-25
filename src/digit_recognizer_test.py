from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import metriculous
import numpy as np
import pandas as pd
import tensorflow as tf

from digit_recognizer import (
    multiply_pixels,
    raw_df_to_x_y,
    read_in_images_as_dataframe,
    save_model_comparison_in_cwd,
    split_into_train_val_test_set,
    stack_to_rgb_image,
)


def test_stack_to_rgb_image() -> None:
    image = np.array([[0, 1, 2], [5, 6, 7]])
    stacked = stack_to_rgb_image(image)
    np.testing.assert_allclose(
        stacked,
        np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[5, 5, 5], [6, 6, 6], [7, 7, 7]]]),
    )
    

def test_split_into_train_and_val_set() -> None:
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    val_split = 0.4
    datasplit = split_into_train_val_test_set(x, y, validation_percentage=val_split)

    print(datasplit.x_train)
    print(datasplit.y_train)
    print(datasplit.x_val)
    print(datasplit.y_val)

    assert len(datasplit.x_train) == 6
    np.testing.assert_allclose(datasplit.x_train, datasplit.y_train)
    np.testing.assert_allclose(datasplit.x_val, datasplit.y_val)


def test_multiply_pixels() -> None:
    image = np.array([[1, 2], [3, 4]])
    new_image = multiply_pixels(image=image, vertical=2, horizontal=3, img_height=2, img_width=2)
    print(new_image)
    np.testing.assert_allclose(
        new_image,
        np.array(
            [
                [1, 1, 1, 2, 2, 2],
                [1, 1, 1, 2, 2, 2],
                [3, 3, 3, 4, 4, 4],
                [3, 3, 3, 4, 4, 4],
            ]
        ),
    )


def test_multiply_pixels_for_multiple_images() -> None:
    images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    new_images = np.array(
        [
            multiply_pixels(image=img, vertical=2, horizontal=3, img_height=2, img_width=2)
            for img in images
        ]
    )

    np.testing.assert_allclose(
        new_images,
        np.array(
            [
                [
                    [1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 3, 3, 4, 4, 4],
                    [3, 3, 3, 4, 4, 4],
                ],
                [
                    [5, 5, 5, 6, 6, 6],
                    [5, 5, 5, 6, 6, 6],
                    [7, 7, 7, 8, 8, 8],
                    [7, 7, 7, 8, 8, 8],
                ],
            ]
        ),
    )


def test_read_in_images_as_dataframe() -> None:
    dir = Path(
        "/Users/x/Desktop/Machine_Learning/digit_recognizer/data/generated_data/"
    )
    dataframe = read_in_images_as_dataframe(dir, 28, 28)
    print(dataframe.shape)

    x, y = raw_df_to_x_y(dataframe)
    print(f"x.shape = {x.shape}")
    print(f"y.shape = {y.shape}")

    data_split = split_into_train_val_test_set(
        x, y, validation_percentage=0.2, test_percentage=0.01
    )

    print(data_split.get_shape())


def test_concat() -> None:
    a = np.array([[0, 1, 2, 3]])
    b = np.array([[0, 11, 22, 33]])
    print(f"np.concatenate(a, b, axis=0) = {np.concatenate((a, b), axis=0)}")


def test_evaluation() -> None:
    ground_truth = np.array([(0, 1, 0), (1, 0, 0), (0, 0, 1)])
    probabilities = np.array([(0.1, 0.9, 0.0), (0.3, 0.7, 0.0), (0.3, 0.7, 0.0)])

    print(f"ground_truth = {ground_truth}")
    print(f"probabilities = {probabilities}")

    class_names = ["a", "b", "c"]
    metriculous.compare_classifiers(
        ground_truth=ground_truth,
        model_predictions=[probabilities],
        class_names=class_names,
    ).save_html("test_comparison.html").display()
