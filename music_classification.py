from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from data_utils import split_dataset, MusicDataset
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor
import os 
import argparse
import json 

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="data/genres")
    args.add_argument("--model_name", type=str, default="microsoft/resnet-50")
    args.add_argument("--cfg", type=str, default="./cfg/training_cfg.json")

    args = args.parse_args()

    music_dataset = MusicDataset(args.data_path)

    feature_extractor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageClassification.from_pretrained(args.model_name, num_labels=len(music_dataset.genres))

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    model_transforms = Compose(
        [RandomResizedCrop(feature_extractor.size), ColorJitter(brightness=0.5, hue=0.5), ToTensor(), normalize]
    )

    music_dataset.transform = model_transforms

    train_dataset, val_dataset, test_dataset = split_dataset(music_dataset)

    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    training_args = TrainingArguments(**cfg)

    # Define the trainer
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    # Train the model
    trainer.train()
