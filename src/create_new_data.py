import os
import time
from pathlib import Path

from PIL import ImageDraw, ImageFont, Image
from digit_recognizer import DATA_DIR


def save_image_with_character(
    width: int, height: int, x: int, y: int, letter: str, font, title: str = "image"
):
    image = Image.new("L", (width, height), (0,))
    draw = ImageDraw.Draw(image)

    draw.text((x, y), letter, font=font, fill=(255,))
    image.save(title)


def main():
    img_height, img_width = 28, 28
    destination_dir = (DATA_DIR / "generated_data")

    Path.mkdir(destination_dir, parents=True)
    os.chdir(destination_dir)

    font_sizes = [16, 18, 20, 22]
    digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    x_coordinates = [4, 6, 8, 10, 12]
    y_coordinates = [1, 3, 5]

    # Path for Fonts on my Mac
    fonts_dir = "/System/Library/Fonts/"

    exclude = [
        "Geeza",
        "Apple Braille",
        "Apple Color",
        "ArabicUI",
        "LastResort",
        "NotoS",
        "NotoN",
        "Savoye",
        "SnellR",
        "ZapfDing",
        "Waseem",
        "ヒラギノ角ゴシック",
    ]
    exclude_for_zero = [
        "Menlo",
        "Monaco",
        "SFNSM",
    ]
    exclude_supplemental_dir = [
        "NISC18030",
        "AlBayan",
        "Andale",
        "Ayuthaya",
        "Baghdad",
        "Bodoni",
        "Damascus",
        "DecoType",
        "Diwan Kufi",
        "Hoefler",
        "Kokonor",
        "Krungthep",
        "Monaco",
        "PartyLET",
        "Phosphate",
        "PTMono",
        "Kailasa",
        "KufiStandard",
        "Nadeem",
        "Webdings",
        "Zapfino",
        "STIXInt",
        "STIXNon",
        "STIXSiz",
        "STIXVar",
        "Wingding",
    ]
    exclude_supplemental_dir_for_zero = ["Silo"]

    for root, directories, files in os.walk(fonts_dir, topdown=True):
        for filename in files:
            if not any(
                filename.startswith(x) for x in (exclude + exclude_supplemental_dir)
            ):
                for font_size in font_sizes:
                    for digit in digits:
                        if digit != "0" or (
                            not any(
                                filename.startswith(x)
                                for x in (
                                    exclude_for_zero + exclude_supplemental_dir_for_zero
                                )
                            )
                        ):
                            for x in x_coordinates:
                                for y in y_coordinates:
                                    font = ImageFont.truetype(
                                        fonts_dir + filename, font_size
                                    )
                                    title = f"class{digit}_{filename}_{font_size}_{x}_{y}.jpg"
                                    save_image_with_character(
                                        width=img_width,
                                        height=img_height,
                                        x=x,
                                        y=y,
                                        letter=str(digit),
                                        font=font,
                                        title=title,
                                    )


if __name__ == "__main__":
    main()
