import sys
import yaml
from utils.dataset import DataModule
from torchvision import transforms
import argparse
from pathlib import Path
from trainers.resnet_trainer import ResNetTrainer
from trainers.cnn_trainer import CNNTrainer
import logging 
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parent.parent))

def setup_logging(log_dir, log_name="train.log"):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging to {log_path}")

def main():
    parser = argparse.ArgumentParser("Model_cfg")
    parser.add_argument("--model", default="resnet_fer", required=True)
    args = parser.parse_args()
    config_path = Path("cfgs") / f"{args.model}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
        data_cfg = cfg["data_args"]

    
    if args.model == "resnet_fer":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
                transforms.RandomRotation(degrees=10),
                transforms.RandomCrop(48, padding=4),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.expand(3, -1, -1)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ]
        )

    elif args.model == "cnn_fer":
        transform = transforms.Compose(
            [   transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally
                transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness/contrast
                transforms.RandomRotation(degrees=10),  # Slight random rotation
                transforms.RandomCrop(48, padding=4),  # Random crop with padding
                transforms.ToTensor(),  # Convert to tensor (if not already)
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),  # RandomErasing is applied to the tensor
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize
            ]
        )
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"model_{args.model}_{now}.log"
    setup_logging(cfg.get("log_dir", "log/"), log_name=log_name)
    dm = DataModule(
        path=data_cfg["path"],
        model_name=args.model,
        transform=transform,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        shuffle=True,
        pin_memory=data_cfg["pin_memory"],
    )

    train_loader = dm.get_train_loader()
    val_loader = dm.get_val_loader()

    if args.model == "resnet_fer":
        trainer = ResNetTrainer(
            config=cfg, 
            train_loader=train_loader,
            eval_loader=val_loader,
        )
    elif args.model == "cnn_fer":
        trainer = CNNTrainer(
            config=cfg,
            train_loader=train_loader,
            eval_loader=val_loader,
        )
    trainer.train()
    trainer.save_model(cfg['train_args']['save_path'])


if __name__ == "__main__":
    main()
