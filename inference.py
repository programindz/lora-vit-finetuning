import torch
import json
from transformers import AutoModelForImageClassification, AutoImageProcessor
from peft import PeftModel


with open('human_action_labels.json', 'r') as f1, open('food_labels.json', 'r') as f2:
    human_action_labels = json.load(f1)
    food_labels = json.load(f2)

human_action_labels = {int(k):v for k, v in human_action_labels.items()}
food_labels = {int(k):v for k, v in food_labels.items()}

human_action_ids = {v:k for k, v in human_action_labels.items()}
food_ids = {v:k for k, v in food_labels.items()}

def inference_model(adaptor):
    model = None
    image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k', use_fast=True)
    if adaptor == 'Food':
        base_model = AutoModelForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            label2id=food_ids,
            id2label=food_labels,
            ignore_mismatched_sizes = True
            )
        model = PeftModel.from_pretrained(base_model, 'lora-food')
        # image_processor = AutoImageProcessor.from_pretrained('lora-food')

    else:
        base_model = AutoModelForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            label2id=human_action_ids,
            id2label=human_action_labels,
            ignore_mismatched_sizes=True
            )
        model = PeftModel.from_pretrained(base_model, 'lora-human')
        # image_processor = AutoImageProcessor.from_pretrained('lora-human')

    return model, image_processor


def predict_image_class(image, model, image_processor):
    encoding = image_processor(image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits

    class_index = logits.argmax(-1).item()
    return model.config.id2label[class_index]