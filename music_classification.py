from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments, AutoConfig
from data_utils import split_dataset, MusicDataset, get_dataloaders
from model_utils import train_from_hf
from torchvision import transforms
import os 
import argparse
import json 
import torch 
import torch.nn as nn

class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''
    
    def __call__(self, image):       
        return image

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="genres")
    args.add_argument("--model_name", type=str, default="microsoft/resnet-50")
    args.add_argument("--cfg", type=str, default="./cfg/training_cfg.json")
    args.add_argument('--fake_rgb', action='store_true')
    args = args.parse_args()

    music_dataset = MusicDataset(os.path.join('data',args.data_path))

    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = music_dataset.genres.shape[0]

    feature_extractor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageClassification.from_config(config)

    print(model)
    if len(feature_extractor.size.values()) == 1:
        feature_extractor.size = {k:(int(v), int(v)) for k,v in feature_extractor.size.items()}
    
    model_transforms = transforms.Compose([
                    transforms.Resize(tuple(feature_extractor.size.values())),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  if args.fake_rgb else NoneTransform(),
                    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    print(model_transforms)
    music_dataset.transform = model_transforms

    #train_dataset, val_dataset, test_dataset = split_dataset(music_dataset)
    train_loader, val_loader, test_loader = get_dataloaders(music_dataset)

    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lr = cfg['lr']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    train_from_hf(model, train_loader, val_loader, optimizer, criterion, device=device, **cfg)