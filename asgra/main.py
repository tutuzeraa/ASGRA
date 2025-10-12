import os
import json
import logging
import argparse
import datetime
import shutil
import random
from pathlib import Path
from operator import itemgetter

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, ConcatDataset
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassRecall, MulticlassF1Score,
    BinaryAccuracy, BinaryRecall, BinaryF1Score
)
from torch_geometric.loader import DataLoader
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold  
import numpy as np

from codecarbon import track_emissions

from datasets.places8 import Places8SceneGraphDataset
from datasets.rcpd import RCPDSceneGraphDataset
from models.GATv2 import ASGRA

DATASETS = {
    "places8": lambda cfg, split, cache: Places8SceneGraphDataset(cfg["data_dir"], cfg["csv_file"], cfg["triplet_threshold"], split, cache),
    "rcpd": lambda cfg, split, cache: RCPDSceneGraphDataset(cfg["data_dir"], split, cache),
}


def set_global_seed(seed: int = 420):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def stamp_dir(base: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return base.parent / f"{ts}_{base.name}"

def setup_logger(logdir: Path, fname: str):
    logdir.mkdir(parents=True, exist_ok=True)
    fmt, datef = "%(asctime)s | %(levelname)s | %(name)s: %(message)s", "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datef, force=True)
    fh = logging.FileHandler(logdir / fname, mode="w")
    fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datef))
    logging.getLogger().addHandler(fh)
    return logging.getLogger(__name__)

def load_cfg(path: Path) -> dict:
    if path.is_file():
        return json.load(open(path))
    raise FileNotFoundError(f"No config file found in {path}")

def get_dataset(cfg, split, cache):
    dataset_name = cfg["dataset"]
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    return DATASETS[dataset_name](cfg, split, cache)

def build_full_dataset(cfg, cache=None):
    splits = ["train", "val", "test"]
    parts = [get_dataset(cfg, s, cache) for s in splits]
    return ConcatDataset(parts)

def get_metrics(cfg, device):
    if cfg["task"] == "binary":
        return BinaryAccuracy().to(device), BinaryRecall().to(device), BinaryF1Score().to(device), 2
    return (
        MulticlassAccuracy(cfg["num_classes"]).to(device),
        MulticlassRecall(cfg["num_classes"], average="macro").to(device),
        MulticlassF1Score(cfg["num_classes"], average="macro").to(device),
        MulticlassRecall(cfg["num_classes"], average="macro").to(device),
    )

def _evaluate_split(model, loader, device, cfg, top_k=10):
    acc, rec, f1, _ = get_metrics(cfg, device)

    all_pred, all_tgt   = [], []
    scored_examples     = []          

    with torch.no_grad():
        for batch in loader:
            batch   = batch.to(device)
            logits  = model(batch)

            if cfg["task"] == "binary":
                prob   = logits.squeeze(1).sigmoid()         
                preds  = (prob > 0.5).int()
                score_vec = prob                            
            else:
                prob   = logits.softmax(1)                   
                score_vec, preds = prob.max(1)              
                score_vec = score_vec                       

            target = batch.y.squeeze()
            all_pred.extend(preds.cpu().tolist())
            all_tgt .extend(target.cpu().tolist())

            acc.update(preds, target)
            rec.update(preds, target)
            f1 .update(preds, target)

            for s, img_id, p, t in zip(score_vec.cpu().tolist(),
                                       batch.image_id,       
                                       preds.cpu().tolist(),
                                       target.cpu().tolist()):
                scored_examples.append((s, img_id, p, t))

    top_examples = sorted(scored_examples, key=itemgetter(0), reverse=True)[:top_k]
    top_ids      = [(e[1], e[0]) for e in top_examples]          

    bacc = balanced_accuracy_score(all_tgt, all_pred)

    return acc.compute().item(), rec.compute().item(), f1.compute().item(), bacc, top_ids

def load_pretrained(model, ckpt_path, mode, logger):
    if not ckpt_path:       
        return
    
    logger.info(f"Finetunning in mode {mode} using pretrained weights from {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    # remove última Linear (8→8) porque agora é 384→1
    for k in ["mlp.3.weight", "mlp.3.bias"]:
        state.pop(k, None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == ["mlp.3.weight", "mlp.3.bias"]

    if mode == "head":
        for n, p in model.named_parameters():
            if not n.startswith("mlp.3"):  
                p.requires_grad = False
   
    # mode == "full"  → não congela nada

# @track_emissions
def run_train(cfg: dict, workers: int, outdir: Path, cache=None, train_override=None, val_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(outdir, "train.log")

    train_ds = train_override or get_dataset(cfg, "train", cache)
    val_ds   = val_override or get_dataset(cfg, "val", cache)

    train_loader = DataLoader(train_ds, cfg["batch_size"], shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, cfg["batch_size"], shuffle=False, num_workers=workers)

    model = ASGRA(
        hidden_dim=cfg["hidden_dim"],
        heads=cfg["heads"],
        num_layers=cfg["num_layers"],
        num_classes=cfg["num_classes"]
    ).to(device)

    load_pretrained(model,
                cfg.get("pretrained_ckpt", ""),
                cfg.get("finetune", "full"),
                logger)
    
    lr = cfg["learning_rate"]
    if cfg["finetune"] == "full":
        print("AQUI")
        lr = 1e-5

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])

    best_f1, wait, patience = 0.0, 0, 15
    best_ckpt = outdir / "results" / "best.pt"
    best_ckpt.parent.mkdir(exist_ok=True)

    for epoch in range(1, cfg.get("max_epochs", 120) + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            loss = F.binary_cross_entropy_with_logits(model(batch).squeeze(1), batch.y.float()) if cfg["task"] == "binary" else \
                   F.cross_entropy(model(batch), batch.y.squeeze())
            optim.zero_grad()
            loss.backward()
            optim.step()

        cur_f1 = _evaluate_split(model, val_loader, device, cfg)[2]

        if cur_f1 > best_f1 + 1e-3:
            best_f1, wait = cur_f1, 0
            torch.save(model.state_dict(), best_ckpt)
            logger.info("Ep %02d NEW BEST F1 = %.4f", epoch, best_f1)
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early-stopping after %d epochs.", epoch)
                break

    logger.info("Training completed.")
    return best_ckpt

def run_eval(cfg, workers, outdir, weights, split, cache=None, split_override=None, return_f1=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(outdir, f"eval_{split}.log")

    eval_ds = split_override or get_dataset(cfg, split, cache)
    eval_loader = DataLoader(eval_ds, cfg["batch_size"], shuffle=False, num_workers=workers)

    model = ASGRA(hidden_dim=cfg["hidden_dim"], heads=cfg["heads"], num_layers=cfg["num_layers"], num_classes=cfg["num_classes"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    acc, rec, f1, bacc, top_preds = _evaluate_split(model, eval_loader, device, cfg)

    # print("IDs with highest score:", top_preds)

    rep_dir = outdir / "reports"       
    rep_dir.mkdir(parents=True, exist_ok=True)

    json.dump({"accuracy": acc, "recall": rec, "f1": f1, "balanced accuracy": bacc}, open(rep_dir / "metrics.json", "w"), indent=2)

    if return_f1:
        return f1


def run_kfold(cfg, workers, outdir, cache=None, k=5):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger  = setup_logger(outdir, "kfold.log")
    full_ds = build_full_dataset(cfg, cache)

    y_all = torch.tensor([full_ds[idx].y.item() for idx in range(len(full_ds))])
    skf   = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results_dir = outdir / "results"
    results_dir.parent.mkdir(exist_ok=True)

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(torch.zeros(len(full_ds)), y_all), 1):
        logger.info(f"=== Fold {fold}/{k}  (train {len(train_idx)}, val {len(val_idx)}) ===")

        ds_train = Subset(full_ds, train_idx)
        ds_val   = Subset(full_ds, val_idx)

        train_loader = DataLoader(ds_train, cfg["batch_size"], shuffle=True,  num_workers=workers)
        val_loader   = DataLoader(ds_val,   cfg["batch_size"], shuffle=False, num_workers=workers)

        model = ASGRA(
            hidden_dim   = cfg["hidden_dim"],
            heads        = cfg["heads"],
            num_layers   = cfg["num_layers"],
            num_classes  = cfg["num_classes"],
        ).to(device)

        if cfg.get("finetune"): 
            load_pretrained(model,
                    cfg.get("pretrained_ckpt", ""),
                    cfg.get("finetune", "full"),
                    logger)
    
        lr = cfg["learning_rate"]
        if cfg["finetune"] == "full":
            lr = 1e-5

        optim = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=cfg["weight_decay"])

        for epoch in range(cfg.get("epochs_per_fold", 20)):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                logits = model(batch)
                loss = (F.binary_cross_entropy_with_logits(logits.squeeze(1), batch.y.float())
                        if cfg["task"] == "binary"
                        else F.cross_entropy(logits, batch.y.squeeze()))
                optim.zero_grad(); loss.backward(); optim.step()

        acc, rec, f1, bacc, top_preds = _evaluate_split(model, val_loader, device, cfg)
        # print("IDs with highest score:", top_preds)
        logger.info(f"Fold {fold}  ACC {acc:.4f}  REC {rec:.4f}  F1 {f1:.4f} Balanced ACC {bacc:.4f}")
        fold_metrics.append((acc, rec, f1, bacc))
        
        fold_ckpt = results_dir / f"fold_{fold}_weights.pt"
        fold_ckpt.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), fold_ckpt)

    accs, recs, f1s, baccs = map(np.array, zip(*fold_metrics))
    logger.info("==== 5-fold report ====")
    logger.info("Accuracy           : %.4f ± %.4f", accs.mean(), accs.std())
    logger.info("Recall             : %.4f ± %.4f", recs.mean(), recs.std())
    logger.info("F1-score           : %.4f ± %.4f", f1s.mean(),  f1s.std())
    logger.info("Balanced Accuracy  : %.4f ± %.4f", baccs.mean(), baccs.std())

    rep_dir = outdir / "reports"; rep_dir.mkdir(exist_ok=True)
    json.dump(
        {"fold_metrics": [dict(acc=float(a), rec=float(r), f1=float(f), bac=float(b))
                          for a, r, f, b in fold_metrics],
         "mean":  dict(acc=float(accs.mean()), rec=float(recs.mean()), f1=float(f1s.mean()), bacc=float(baccs.mean())),
         "std":   dict(acc=float(accs.std()),  rec=float(recs.std()),  f1=float(f1s.std()), bacc=float(baccs.std()))},
        open(rep_dir / "kfold_metrics.json", "w"), indent=2
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("-o", "--outdir", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()

    exp_dir = stamp_dir(args.outdir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, exp_dir / args.config.name)

    cfg = load_cfg(args.config)

    set_global_seed(cfg.get("seed", 42))

    if args.mode == "train":
        ckpt = run_train(cfg, args.workers, exp_dir, args.cache)
        run_eval(cfg, args.workers, exp_dir, ckpt, "val", args.cache)
    elif args.mode in ["eval", "test"]:
        if not args.weights:
            raise ValueError("--weights is required")
        split = "val" if args.mode == "eval" else "test"
        run_eval(cfg, args.workers, exp_dir, args.weights, split, args.cache)
    elif args.mode == "xval":
        run_kfold(cfg, args.workers, exp_dir, args.cache, k=5)
