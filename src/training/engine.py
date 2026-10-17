import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.metrics import build_evaluation_result


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, class_weights, scheduler=None, grad_clip=1.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}")
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        loss_sum += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix({"loss": f"{loss_sum / total:.4f}", "acc": f"{correct / total:.4f}"})
    return loss_sum / total, correct / total


def evaluate(model, loader, device, class_names, split: str):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            loss_sum += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_preds_np = torch.cat(all_preds).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    all_probs_np = torch.cat(all_probs).numpy()
    return build_evaluation_result(
        split=split,
        loss=loss_sum / total,
        labels=all_labels_np,
        preds=all_preds_np,
        probs=all_probs_np,
        class_names=class_names,
    )
