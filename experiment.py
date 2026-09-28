"""MNIST kNN course experiment.

Teacher source: IMUEEMM/DL-CLASS, pj1-KNN/KNN_student.py, commit
feec5b1b1bb93f7b70b369c74fd98a1bf7eaab91. This is a completed,
independent implementation of its TODOs; the unmodified source is kept beside it.
Run `python experiment.py explore` in base, then
`conda run -n pytorch311 python experiment.py final` for the full evaluation.
"""
from __future__ import annotations

import json
import platform
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 42
K_VALUES = (1, 3, 5, 7, 9)


def load_mnist():
    """Read the four canonical IDX entries; ZIP contains duplicate copies."""
    names = ("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
             "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    with zipfile.ZipFile(ROOT / "KNN.zip") as z:
        blobs = [z.read(n) for n in names]
    def images(raw):
        magic, n, h, w = struct.unpack(">IIII", raw[:16])
        assert magic == 2051 and (h, w) == (28, 28)
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, h*w).copy()
    def labels(raw):
        magic, n = struct.unpack(">II", raw[:8])
        assert magic == 2049
        return np.frombuffer(raw, np.uint8, offset=8).copy()
    xtr, ytr, xte, yte = images(blobs[0]), labels(blobs[1]), images(blobs[2]), labels(blobs[3])
    assert xtr.shape == (60000, 784) and xte.shape == (10000, 784)
    assert len(ytr) == 60000 and len(yte) == 10000
    return xtr, ytr, xte, yte


def balanced_indices(y, per_class, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False)
                                   for c in range(10)]))


def pca_fit_transform(xtrain, xother, components=50):
    """Training-only centering and eigen decomposition of covariance (no sklearn PCA)."""
    a = xtrain.astype(np.float32) / 255
    b = xother.astype(np.float32) / 255
    mean = a.mean(axis=0)
    centered = a - mean
    cov = (centered.T @ centered) / (len(a) - 1)
    eigenvalues, vectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1][:components]
    basis = vectors[:, order]
    ratio = float(eigenvalues[order].sum() / eigenvalues.clip(min=0).sum())
    return np.ascontiguousarray(centered @ basis), np.ascontiguousarray((b-mean) @ basis), ratio


def hog_features(x):
    from skimage.feature import hog
    return np.asarray([hog(row.reshape(28, 28), orientations=9,
                           pixels_per_cell=(7, 7), cells_per_block=(2, 2),
                           block_norm="L2-Hys", feature_vector=True) for row in x], dtype=np.float32)


def neighbors_numpy(xtrain, xquery, max_k=9, metric="l2", batch=64):
    """Batched vectorized distances; never allocate the full test x train matrix."""
    a = np.ascontiguousarray(xtrain, dtype=np.float32)
    b = np.ascontiguousarray(xquery, dtype=np.float32)
    result_i, result_d = [], []
    norms = (a*a).sum(axis=1) if metric == "l2" else None
    for start in range(0, len(b), batch):
        q = b[start:start+batch]
        if metric == "l2":
            d = np.maximum((q*q).sum(axis=1, keepdims=True) + norms[None, :] - 2*q@a.T, 0)
        else:
            d = np.abs(q[:, None, :] - a[None, :, :]).sum(axis=2)
        part = np.argpartition(d, max_k-1, axis=1)[:, :max_k]
        order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1, kind="stable")
        idx = np.take_along_axis(part, order, axis=1)
        result_i.append(idx)
        result_d.append(np.take_along_axis(d, idx, axis=1))
    return np.vstack(result_i), np.vstack(result_d)


def neighbors_torch(xtrain, xquery, max_k=9, batch=128):
    """Same squared L2 formula on CUDA for the full 60k x 10k evaluation."""
    import torch
    assert torch.cuda.is_available(), "Full evaluation needs CUDA for practical runtime"
    device = "cuda"
    a = torch.as_tensor(np.asarray(xtrain, dtype=np.float32), device=device)
    a2 = (a*a).sum(1)
    result_i, result_d = [], []
    for start in range(0, len(xquery), batch):
        b = torch.as_tensor(np.asarray(xquery[start:start+batch], dtype=np.float32), device=device)
        d = ((b*b).sum(1)[:, None] + a2[None, :] - 2*b@a.T).clamp_min_(0)
        vals, inds = torch.topk(d, max_k, dim=1, largest=False, sorted=True)
        result_i.append(inds.cpu().numpy())
        result_d.append(vals.cpu().numpy())
    torch.cuda.synchronize()
    return np.vstack(result_i), np.vstack(result_d)


def vote(labels, distances, k, weighted=False):
    """Class score desc, class index asc on ties; zero-distance votes dominate."""
    lab, dis = labels[:, :k], distances[:, :k]
    scores = np.zeros((len(lab), 10), dtype=np.float64)
    if weighted:
        zero = dis <= 1e-8
        weights = np.where(zero.any(axis=1, keepdims=True), zero.astype(float),
                           1 / np.maximum(dis, 1e-8))
    else:
        weights = np.ones_like(dis, dtype=float)
    for c in range(10):
        scores[:, c] = ((lab == c)*weights).sum(axis=1)
    ranking = np.argsort(-scores, axis=1, kind="stable")
    return ranking[:, 0], ranking


def evaluate(ytrue, labels, distances, k, weighted=False):
    pred, rank = vote(labels, distances, k, weighted)
    return {"top1": float(np.mean(pred == ytrue)),
            "top3": float(np.mean((rank[:, :3] == ytrue[:, None]).any(axis=1))),
            "top5": float(np.mean((rank[:, :5] == ytrue[:, None]).any(axis=1)))}


def measure(xtr, ytr, xval, yval, metric="l2", engine="numpy", batch=64):
    t0 = time.perf_counter()
    # kNN fit only stores data. Copy here to measure a concrete, consistent operation.
    model_x = np.ascontiguousarray(xtr, dtype=np.float32)
    model_y = ytr.copy()
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if engine == "torch":
        ni, nd = neighbors_torch(model_x, xval, batch=batch)
    else:
        ni, nd = neighbors_numpy(model_x, xval, metric=metric, batch=batch)
    predict_s = time.perf_counter() - t0
    labs = model_y[ni]
    return labs, nd, {"fit_s": fit_s, "neighbors_s": predict_s}


def explore():
    from sklearn.neighbors import KNeighborsClassifier
    xtr, ytr, xte, yte = load_mnist()
    train_idx = balanced_indices(ytr, 1200)
    remaining = np.setdiff1d(np.arange(len(ytr)), train_idx)
    val_idx = balanced_indices(ytr[remaining], 200)
    val_idx = remaining[val_idx]
    xt, yt, xv, yv = xtr[train_idx], ytr[train_idx], xtr[val_idx], ytr[val_idx]
    record = {"seed": SEED, "train_count": len(yt), "validation_count": len(yv),
              "test_count": len(yte), "class_train": np.bincount(ytr).tolist(),
              "class_test": np.bincount(yte).tolist(), "python": sys.version,
              "platform": platform.platform(), "numpy": np.__version__}
    t0 = time.perf_counter()
    raw_train, raw_val = xt.astype(np.float32)/255, xv.astype(np.float32)/255
    record["pixel_transform_s"] = time.perf_counter()-t0
    # Teacher's sklearn baseline, same training/validation data as handmade kNN.
    t0 = time.perf_counter()
    clf = KNeighborsClassifier(n_neighbors=5, algorithm="brute", metric="euclidean", n_jobs=-1)
    clf.fit(raw_train, yt)
    fit_s = time.perf_counter()-t0
    t0 = time.perf_counter()
    baseline = clf.predict(raw_val)
    pred_s = time.perf_counter()-t0
    record["sklearn_baseline"] = {"k":5,"top1":float(np.mean(baseline==yv)),"fit_s":fit_s,"predict_s":pred_s}
    labs, dist, timing = measure(raw_train, yt, raw_val, yv, batch=64)
    record["raw_timing"] = timing
    record["k_scan"] = {str(k): evaluate(yv,labs,dist,k) for k in K_VALUES}
    manual, _ = vote(labs,dist,5)
    record["agreement_sklearn"] = float(np.mean(manual==baseline))
    assert record["agreement_sklearn"] > .995
    record["weighted"] = {str(k): evaluate(yv,labs,dist,k,True) for k in K_VALUES}
    t0 = time.perf_counter()
    pca_train,pca_val,ratio = pca_fit_transform(xt,xv,50)
    pca_s = time.perf_counter()-t0
    pl,pd,pt = measure(pca_train,yt,pca_val,yv,batch=128)
    record["pca"]={"components":50,"variance_ratio":ratio,"transform_s":pca_s,
                   "timing":pt,"k_scan":{str(k):evaluate(yv,pl,pd,k) for k in K_VALUES}}
    t0 = time.perf_counter()
    hog_train,hog_val=hog_features(xt),hog_features(xv)
    hog_s=time.perf_counter()-t0
    hl,hd,ht=measure(hog_train,yt,hog_val,yv,batch=64)
    record["hog"]={"features":int(hog_train.shape[1]),"transform_s":hog_s,
                   "timing":ht,"k_scan":{str(k):evaluate(yv,hl,hd,k) for k in K_VALUES}}
    # L1 is expensive in 784 dimensions; use the same 1200/200 per-class split
    # after PCA reduction, preserving fair data volume for the L1/L2 comparison.
    ll,ld,lt=measure(pca_train,yt,pca_val,yv,metric="l1",batch=8)
    record["pca_l1"]={"timing":lt,"k_scan":{str(k):evaluate(yv,ll,ld,k) for k in K_VALUES}}
    record["validation_indices"]={"train":train_idx.tolist(),"validation":val_idx.tolist()}
    (OUT/"explore.json").write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in record.items() if k!="validation_indices"},ensure_ascii=False,indent=2))


def final():
    xtr,ytr,xte,yte=load_mnist()
    # Chosen from validation results, not from the test set.
    config=json.loads((OUT/"chosen.json").read_text(encoding="utf-8"))
    k=int(config["k"]); weighted=bool(config["weighted"])
    t0=time.perf_counter()
    if config["feature"]=="pca50":
        a,b,variance_ratio=pca_fit_transform(xtr,xte,50)
        feature="PCA 50 components from normalized raw pixels"
    else:
        a=xtr.astype(np.float32)/255
        b=xte.astype(np.float32)/255
        variance_ratio=None
        feature="raw pixels / 255"
    transform_s=time.perf_counter()-t0
    labs,dist,timing=measure(a,ytr,b,yte,engine="torch",batch=128)
    pred,rank=vote(labs,dist,k,weighted)
    cm=np.zeros((10,10),dtype=int)
    np.add.at(cm,(yte,pred),1)
    chosen=balanced_indices(yte,10)
    result={"train_count":len(ytr),"test_count":len(yte),"k":k,"weighted":weighted,
            "feature":feature,"distance":"squared L2", "timing":timing,
            "transform_s":transform_s,"variance_ratio":variance_ratio,
            "metrics":evaluate(yte,labs,dist,k,weighted),"cm":cm.tolist(),
            "per_class_accuracy":(np.diag(cm)/cm.sum(axis=1)).tolist(),
            "samples":[{"test_index":int(i),"truth":int(yte[i]),"pred":int(pred[i])} for i in chosen],
            "sample_correct_by_class":[int(np.sum(pred[chosen[yte[chosen]==c]]==c)) for c in range(10)]}
    (OUT/"final.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    np.savez_compressed(OUT/"sample_images.npz",images=xte[chosen].reshape(-1,28,28),truth=yte[chosen],pred=pred[chosen])
    print(json.dumps({k:v for k,v in result.items() if k!="samples"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ("explore","final"):
        raise SystemExit("usage: experiment.py explore|final")
    globals()[sys.argv[1]]()
