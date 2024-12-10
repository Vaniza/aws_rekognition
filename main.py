from pathlib import Path
import boto3  # type: ignore
from mypy_boto3_rekognition.type_defs import (  # type: ignore
    CelebrityTypeDef,
    RecognizeCelebritiesResponseTypeDef,
)
from PIL import Image, ImageDraw, ImageFont  # type: ignore

rekognition_client = boto3.client("rekognition")


def build_image_path(file_name: str) -> str:
    return str(Path(__file__).parent / "images" / file_name)


def detect_celebrities(image_path: str) -> RecognizeCelebritiesResponseTypeDef:
    with open(image_path, "rb") as img_file:
        return rekognition_client.recognize_celebrities(Image={"Bytes": img_file.read()})


def annotate_image(
    input_image_path: str,
    output_image_path: str,
    celebrity_faces: list[CelebrityTypeDef],
):
    img = Image.open(input_image_path)
    drawer = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Ubuntu-R.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    img_width, img_height = img.size

    for celebrity in celebrity_faces:
        bounding_box = celebrity["Face"]["BoundingBox"]
        x1 = int(bounding_box["Left"] * img_width)
        y1 = int(bounding_box["Top"] * img_height)
        x2 = int((bounding_box["Left"] + bounding_box["Width"]) * img_width)
        y2 = int((bounding_box["Top"] + bounding_box["Height"]) * img_height)

        match_score = celebrity.get("MatchConfidence", 0)
        if match_score > 90:
            drawer.rectangle([x1, y1, x2, y2], outline="blue", width=3)

            name = celebrity.get("Name", "Desconhecido")
            text_position = (x1, y1 - 20)
            text_bbox = drawer.textbbox(text_position, name, font=font)
            drawer.rectangle(text_bbox, fill="blue")
            drawer.text(text_position, name, font=font, fill="white")

    img.save(output_image_path)
    print(f"Imagem anotada salva em: {output_image_path}")


if __name__ == "__main__":
    image_files = [
        build_image_path("bbc.jpg"),
        build_image_path("msn.jpg"),
        build_image_path("neymar-torcedores.jpg"),
    ]

    for img_path in image_files:
        result = detect_celebrities(img_path)
        celebrities = result["CelebrityFaces"]
        if not celebrities:
            print(f"Nenhuma celebridade reconhecida na imagem: {img_path}")
            continue

        output_path = build_image_path(f"{Path(img_path).stem}-annotated.jpg")
        annotate_image(img_path, output_path, celebrities)
