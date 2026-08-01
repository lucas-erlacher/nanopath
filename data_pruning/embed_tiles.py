import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from model import DinoV2ViT, load_dinov2_pretrained
from utils import assert_shape

SHARD_LIMIT = 1


@torch.no_grad()
def embed_tiles(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(cfg["train"]["global_size"])
    
    backbone = build_embedding_model(cfg).to(device).eval()

    to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
        
    # batch_dim, rgb_dim, height, width
    mean_t = torch.tensor(cfg["data"]["mean"], device=device).view(1, 3, 1, 1)  # insert singleton dims to enable broadcasting 
    std_t = torch.tensor(cfg["data"]["std"], device=device).view(1, 3, 1, 1)

    paths, batch, embeddings = [], [], []

    dataset_dir = Path(cfg["data"]["dataset_dir"])
    shard_paths = sorted(dataset_dir.glob("shard-*.parquet"))
    
    if not shard_paths:
        raise RuntimeError(f"no parquet shards found under {dataset_dir}.")
   
    for shard_path in tqdm(shard_paths[:SHARD_LIMIT], desc="embedding shards"):
        table = pq.read_table(str(shard_path), columns=["path", "jpeg"])
            shard_rows = zip(table["path"].to_pylist(), table["jpeg"].to_pylist())
            for path, jpeg in tqdm(shard_rows, total=table.num_rows, desc=shard_path.name, leave=False):
            batch.append(to_tensor(Image.open(io.BytesIO(jpeg)).convert("RGB")))
            paths.append(path)

            if len(batch) == cfg["prune"]["embedding_model_batchsize"]:
                embeddings.append(embed_batch(batch, device, backbone, mean_t, std_t, image_size))
                batch = []

    if batch:
        embeddings.append(embed_batch(batch, device, backbone, mean_t, std_t, image_size))    
    
    return paths, np.concatenate(embeddings, axis=0)

def build_embedding_model(cfg):
    model = cfg["prune"]["embedding_model"]

    if model == "dinov2_vits14_reg":
        return load_dinov2_pretrained(DinoV2ViT(variant="dinov2_vits14_reg"))
    else:
        raise RuntimeError(f"configured model {model} is not supported")

def embed_batch(batch, device, backbone, mean_t, std_t, image_size):
    x = torch.stack(batch).to(device)
    assert_shape(x, ["*", 3, image_size, image_size], "stacked batch")

    expected_shape = tuple(x.shape)
    x = (x - mean_t) / std_t  # uses broadcasting to apply mean and std to every pixel in every image in batch
    assert_shape(x, expected_shape, "preprocessed batch")
    
    return backbone.probe_features(x).float().cpu().numpy()


def main():
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())

    paths, embeddings = embed_tiles(cfg)

    embed_path = Path(cfg["prune"]["embeddings_path"]).with_suffix(".embeddings.npz")
    embed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embed_path, paths=np.asarray(paths), embeddings=embeddings.astype(np.float32))

    print(f"saved embeddings to {embed_path}")


if __name__ == "__main__":
    main()