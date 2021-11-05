import numpy as np
import metriculous
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tensorflow as tf

from digit_recognizer import (
    stack_to_rgb_image,
    save_predictions_for_submission,
    split_into_train_val_test_set,
    multiply_pixels,
    raw_df_to_x_y,
    perform_data_augmentation,
    read_in_images_as_dataframe,
    save_model_comparison_in_cwd,
)


def test_stack_to_rgb_image():
    image = np.array([[0, 1, 2], [5, 6, 7]])
    stacked = stack_to_rgb_image(image)
    np.testing.assert_allclose(
        stacked,
        np.array(
            [[[0, 0, 0], [1, 1, 1], [2, 2, 2]], [[5, 5, 5], [6, 6, 6], [7, 7, 7]]]
        ),
    )


def test_save_predictions_for_submission():
    path = "../predictions/"
    predictions = pd.DataFrame([[1, 2, 3], [6, 3, 1], [1, 9, 1]])
    save_predictions_for_submission(predictions=predictions, path_to_dir=path)


def test_split_into_train_and_val_set():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    val_split = 0.6
    datasplit = split_into_train_val_test_set(x, y, validation_split=val_split)

    print(datasplit.x_train)
    print(datasplit.y_train)
    print(datasplit.x_val)
    print(datasplit.y_val)

    assert len(datasplit.x_train) == 6
    np.testing.assert_allclose(datasplit.x_train, datasplit.y_train)
    np.testing.assert_allclose(datasplit.x_val, datasplit.y_val)


def test_multiply_pixels():
    image = np.array([[1, 2], [3, 4]])
    new_image = multiply_pixels(image=image, vertical=2, horizontal=3)
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


def test_multiply_pixels_for_multiple_images():
    images = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    new_images = np.array(
        [
            multiply_pixels(
                image=img, vertical=2, horizontal=3, img_height=2, img_width=2
            )
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


def test_perform_data_augmentation():
    df_img = pd.read_csv((Path(__file__).parent.parent / "data/train_sample.csv"))
    x_img, y_img = raw_df_to_x_y(raw_data_frame=df_img)
    plt.imshow(x_img[0])
    plt.show()

    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10)
    iterator = data_generator.flow(x_img, y_img)

    for i in range(2):
        # define subplot
        plt.subplot(330 + 1 + i)
        # generate batch of images
        batch = iterator.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype("uint8")
        # plot raw pixel data
        plt.imshow(batch[0])


def test_read_in_images_as_dataframe():
    dir = Path(
        "/Users/x/Desktop/Machine_Learning/digit_recognizer/data_generated/test_data_generated"
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


def test_concat():
    a = np.array([[0, 1, 2, 3]])
    b = np.array([[0, 11, 22, 33]])
    print(f"np.concatenate(a, b, axis=0) = {np.concatenate((a, b), axis=0)}")


def test_evaluation():
    ground_truth = np.array([(0, 1, 0), (1, 0, 0), (0, 0, 1)])
    probabilities = np.array([(0.1, 0.9, 0.0), (0.3, 0.7, 0.0), (0.3, 0.7, 0.0)])

    print(f"ground_truth = {ground_truth}")
    print(f"probabilities = {probabilities}")

    class_names = ["a", "b", "c"]
    metriculous.compare_classifiers(
        ground_truth=ground_truth,
        model_predictions=[probabilities],
        class_names=class_names,
    ).save_html("comparison.html").display()
