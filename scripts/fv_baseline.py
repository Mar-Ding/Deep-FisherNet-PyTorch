"""FV baseline: Stage1 backbone + sklearn GMM + SVM (no Fisher layer training)."""
from __future__ import annotations
import argparse, sys, math, numpy as np, torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as TF
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fishernet.configs import PRESETS, apply_preset
from fishernet.data.voc import PASCAL_CLASSES, VOCClassification, collate_voc_batch
from fishernet.data.patches import make_dense_patch_boxes
from fishernet.models import build_fishernet
from fishernet.utils import mean_average_precision

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="vgg16-paper-like")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--stage1-weights", type=Path, required=True)
    p.add_argument("--num-components", type=int, default=32)
    p.add_argument("--max-descriptors", type=int, default=200000)
    p.add_argument("--svm-C", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    return apply_preset(args)

@torch.no_grad()
def collect_descriptors(model, loader, device, max_desc):
    """Extract patch descriptors from training images using backbone only."""
    model.eval()
    all_desc = []
    total = 0
    for batch in tqdm(loader, desc="collect descriptors"):
        images = batch["images"].to(device)
        boxes = [b.to(device) for b in batch["boxes"]]
        descriptors, mask = model.extract_patch_features(images, boxes)
        valid = descriptors[mask].detach().cpu()
        all_desc.append(valid)
        total += valid.shape[0]
        if total >= max_desc:
            break
    return torch.cat(all_desc, dim=0)[:max_desc].numpy()

@torch.no_grad()
def encode_fv(model, loader, device, gmm: GaussianMixture, multi_scale: bool):
    """Encode all images into FV features using sklearn GMM."""
    from fishernet.data.patches import make_dense_patch_boxes
    from fishernet.data.voc import load_and_preprocess
    
    model.eval()
    all_fv, all_labels = [], []
    
    # Determine scales from preset or default
    scales = [480, 576, 688, 864, 1200] if multi_scale else [448]
    boxes = make_dense_patch_boxes(448, 448, 
        patch_sizes=(64, 96, 128, 160, 192, 224, 256), stride=32, max_patches=800)
    
    K, D = gmm.n_components, gmm.means_.shape[1]
    precisions = 1.0 / np.sqrt(gmm.covariances_)
    
    for batch in tqdm(loader, desc="encode FV"):
        images = batch["images"].to(device)
        labels_batch = batch["labels"]
        boxes_b = [b.to(device) for b in batch["boxes"]]
        
        descriptors, mask = model.extract_patch_features(images, boxes_b)
        desc_np = descriptors[mask].cpu().numpy()
        
        # Compute FV using sklearn GMM
        # gamma: softmax assignments
        gamma = gmm.predict_proba(desc_np)  # [M, K]
        
        # First order: gamma * (x - mu) / sigma
        diff = desc_np[:, None, :] - gmm.means_[None, :, :]  # [M, K, D]
        scaled_diff = diff * precisions[None, :, :]  # [M, K, D]
        first = (gamma[:, :, None] * scaled_diff).sum(axis=0) / desc_np.shape[0]  # [K, D]
        
        # Second order: gamma * ((x-mu)^2/sigma^2 - 1) / sqrt(2)
        scaled_diff_sq = scaled_diff ** 2
        second = (gamma[:, :, None] * (scaled_diff_sq - 1.0)) / math.sqrt(2)
        second = second.sum(axis=0) / desc_np.shape[0]
        
        fv = np.concatenate([first.ravel(), second.ravel()])
        all_fv.append(fv)
        all_labels.append(labels_batch[0].numpy())
    
    return np.stack(all_fv), np.stack(all_labels)

def main():
    args = parse_args()
    device = args.device
    
    # Build model with Stage 1 backbone only
    model = build_fishernet(
        backbone=args.backbone, num_classes=len(PASCAL_CLASSES),
        patch_dim=args.patch_dim, num_components=args.num_components,
        pretrained=False, roi_output_size=args.roi_output_size,
    ).to(device)
    
    ckpt = torch.load(args.stage1_weights, map_location=device)
    sd = ckpt.get("model", ckpt)
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k.startswith("features.") and k in model_sd and v.shape == model_sd[k].shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd)
    print(f"Loaded {loaded} feature layers from Stage 1")
    
    # Data
    train_ds = VOCClassification(root=args.data_root, year="2007", image_set="trainval",
        image_size=448, train_scales=None, hflip_prob=0.0,
        patch_sizes=(64, 96, 128, 160, 192, 224, 256), patch_stride=32, max_patches=800)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_voc_batch)
    
    test_ds = VOCClassification(root=args.data_root, year="2007", image_set="test",
        image_size=448, train_scales=None, hflip_prob=0.0,
        patch_sizes=(64, 96, 128, 160, 192, 224, 256), patch_stride=32, max_patches=800)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_voc_batch)
    
    # Step 1: Collect descriptors & fit GMM
    print("Collecting descriptors...")
    descriptors = collect_descriptors(model, train_loader, device, args.max_descriptors)
    print(f"  {descriptors.shape[0]} descriptors, dim={descriptors.shape[1]}")
    
    print("Fitting GMM...")
    gmm = GaussianMixture(n_components=args.num_components, covariance_type="diag",
                          max_iter=100, random_state=7, verbose=1)
    gmm.fit(descriptors)
    print(f"  GMM done: K={gmm.n_components}")
    
    # Step 2: Encode train & test FV
    print("Encoding train FV...")
    train_fv, train_labels = encode_fv(model, train_loader, device, gmm, multi_scale=True)
    print(f"  Train FV: {train_fv.shape}")
    
    print("Encoding test FV...")
    test_fv, test_labels = encode_fv(model, test_loader, device, gmm, multi_scale=True)
    print(f"  Test FV: {test_fv.shape}")
    
    # Step 3: Power norm + L2 norm
    train_fv = np.sign(train_fv) * np.sqrt(np.abs(train_fv) + 1e-6)
    train_fv = train_fv / (np.linalg.norm(train_fv, axis=1, keepdims=True) + 1e-6)
    test_fv = np.sign(test_fv) * np.sqrt(np.abs(test_fv) + 1e-6)
    test_fv = test_fv / (np.linalg.norm(test_fv, axis=1, keepdims=True) + 1e-6)
    
    scaler = StandardScaler()
    train_fv = scaler.fit_transform(train_fv)
    test_fv = scaler.transform(test_fv)
    
    # Step 4: Train SVM per class
    print("Training SVMs...")
    svm_models = []
    for c in range(train_labels.shape[1]):
        y_bin = train_labels[:, c]
        if y_bin.max() == 0:
            svm_models.append(None)
            continue
        svm = LinearSVC(C=args.svm_C, max_iter=10000, dual="auto", random_state=7)
        svm.fit(train_fv, y_bin)
        svm_models.append(svm)
    
    # Step 5: Evaluate
    all_scores = []
    for c, svm in enumerate(svm_models):
        if svm is None:
            all_scores.append(np.zeros(test_fv.shape[0]))
        else:
            all_scores.append(svm.decision_function(test_fv))
    scores = np.stack(all_scores, axis=1)
    
    mean_ap, aps = mean_average_precision(
        torch.from_numpy(test_labels).float(),
        torch.from_numpy(scores).float()
    )
    print(f"\n=== FV Baseline mAP: {mean_ap:.4f} ===")
    for name, ap in zip(PASCAL_CLASSES, aps):
        print(f"  {name:12s} {ap:.4f}")

if __name__ == "__main__":
    main()
