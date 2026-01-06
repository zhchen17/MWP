# train.py
import torch
from torch import optim
from tqdm import tqdm

from models.encoder import MWPEncoder
from models.wpe import WarmPriorEstimator
from models.gmr import GranularMultiRetriever
from models.framework import MWPFramework
from dataprocess import get_dataloaders


def main(args):
    device = torch.device(args.device)

    encoder = MWPEncoder(args.clip_path, device)
    wpe = WarmPriorEstimator(device=device)
    gmr = GranularMultiRetriever(device=device)

    model = MWPFramework(
        encoder, wpe, gmr,
        lambda_prior=0.5
    ).to(device)

    optimizer = optim.AdamW(
        list(encoder.parameters()) +
        list(gmr.parameters()),
        lr=1e-5
    )

    train_loader, _, _ = get_dataloaders(args)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            loss = model(
                batch["caption_input_ids"].to(device),
                batch["caption_attention_mask"].to(device),
                batch["pixel_values"].to(device),
                batch["image_description_input_ids"].to(device),
                batch["image_description_attention_mask"].to(device),
                batch["category"].to(device),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: {total_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_json", type=str, default="./train.json")
    parser.add_argument("--val_json", type=str, default="./val.json")
    parser.add_argument("--test_json", type=str, default="./val.json")
    parser.add_argument("--image_root", type=str, default="./image_all_mini")
    parser.add_argument("--gpu_ids", type=str, default="4")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--text_max_length", type=int, default=77)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug_ratio", type=float, default=1.0)
    args = parser.parse_args()
    args = parser.parse_args()

    main(args)
