import albumentations as A
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from transformers import pipeline


# Mapping for labels and IDs
id2label = {
    0: 'shirt, blouse', 1: 'top, t-shirt, sweatshirt', 2: 'sweater', 3: 'cardigan',
    4: 'jacket', 5: 'vest', 6: 'pants', 7: 'shorts', 8: 'skirt', 9: 'coat',
    10: 'dress', 11: 'jumpsuit', 12: 'cape', 13: 'glasses', 14: 'hat',
    15: 'headband, head covering, hair accessory', 16: 'tie', 17: 'glove',
    18: 'watch', 19: 'belt', 20: 'leg warmer', 21: 'tights, stockings',
    22: 'sock', 23: 'shoe', 24: 'bag, wallet', 25: 'scarf', 26: 'umbrella',
    27: 'hood', 28: 'collar', 29: 'lapel', 30: 'epaulette', 31: 'sleeve',
    32: 'pocket', 33: 'neckline', 34: 'buckle', 35: 'zipper', 36: 'applique',
    37: 'bead', 38: 'bow', 39: 'flower', 40: 'fringe', 41: 'ribbon',
    42: 'rivet', 43: 'ruffle', 44: 'sequin', 45: 'tassel'
}

label2id = {v: k for k, v in id2label.items()}

# Training data transformations
def get_train_transform():
    return A.Compose(
        [
            A.LongestMaxSize(500),
            A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.GaussianBlur(p=0.5),
            A.GaussNoise(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"]
        ),
    )

# Validation data transformations
def get_val_transform():
    return A.Compose(
        [
            A.LongestMaxSize(500),
            A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category"]
        ),
    )

# Load the model
def load_model(device: int = -1):
    """
    Load the DETR model pipeline.
    :param device: Specify device to load the model (-1 for CPU, 0 for GPU).
    :return: Hugging Face object-detection pipeline.
    """
    model_pipeline = pipeline(
        "object-detection", 
        model="sergiopaniego/detr-resnet-50-dc5-fashionpedia-finetuned", 
        device=device
    )
    return model_pipeline
