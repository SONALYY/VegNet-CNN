# train_pytorch.py
import argparse, os, json, random, time, platform
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loaders(data_dir, img_size=224, batch_size=32, val_split=0.2):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_ds = datasets.ImageFolder(data_dir, transform=train_tfms)
    class_names = full_ds.classes

    val_size = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    # use val transforms for the val split
    val_ds.dataset.transform = val_tfms

    # Windows: keep workers=0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, class_names

def build_model(num_classes, pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total,1), correct / max(total,1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Folder that contains class subfolders")
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_only", action="store_true",
                   help="If set, only train the final layer (no later fine-tune).")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = build_loaders(
        args.data_dir, img_size=args.img_size, batch_size=args.batch_size, val_split=args.val_split
    )

    model = build_model(num_classes=len(class_names)).to(device)

    # ✅ Speed-up on CPU: train only the final layer first
    for name, p in model.named_parameters():
        if "fc" not in name:
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump({"class_names": class_names}, f, indent=2)

    best_acc = 0.0
    history = []

    try:
        # Stage 1: train final layer only
        for epoch in range(1, args.epochs+1):
            tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
            vl, va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
            history.append({"stage":"head", "epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
            print(f"[HEAD] Epoch {epoch:02d} | train_loss={tl:.4f} acc={ta:.3f} || val_loss={vl:.4f} acc={va:.3f}")
            if va > best_acc:
                best_acc = va
                torch.save(model.state_dict(), os.path.join(args.out_dir, "vegnet_resnet18.pth"))
                print(f"  ✔ Saved new best (val_acc={best_acc:.3f})")

        if not args.freeze_only:
            # Stage 2: light fine-tune top layers (last block + fc)
            for name, p in model.named_parameters():
                p.requires_grad = False
            # Unfreeze last ~layer4 + fc
            for name, p in list(model.named_parameters()):
                if name.startswith("layer4") or name.startswith("fc"):
                    p.requires_grad = True

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=max(args.lr*0.5, 5e-4))
            ft_epochs = max(2, args.epochs // 4)  # short fine-tune

            for epoch in range(1, ft_epochs+1):
                tl, ta = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
                vl, va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
                history.append({"stage":"finetune", "epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
                print(f"[FT  ] Epoch {epoch:02d} | train_loss={tl:.4f} acc={ta:.3f} || val_loss={vl:.4f} acc={va:.3f}")
                if va > best_acc:
                    best_acc = va
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "vegnet_resnet18.pth"))
                    print(f"  ✔ Saved new best (val_acc={best_acc:.3f})")

    except KeyboardInterrupt:
        print("Stopping early; saving last checkpoint...")

    finally:
        # Always save last + history
        torch.save(model.state_dict(), os.path.join(args.out_dir, "vegnet_resnet18_last.pth"))
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"Done. Best val_acc={best_acc:.3f}. Artifacts in: {args.out_dir}")

if __name__ == "__main__":
    main()
