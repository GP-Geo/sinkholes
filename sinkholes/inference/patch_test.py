"""Patch-level test: a pickled test split + a checkpoint -> Dice, pixel and
object-level precision/recall.

Run as ``sinkholes test-patches``. Reports the mean of per-patch Dice (macro —
the harsher number; ``inspect-run`` reports the pooled micro variant).
"""

import argparse
import logging


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--test_data_path", type=str, required=True,
                   help="pickled test split written by training")
    p.add_argument("--model", type=str, required=True, help="checkpoint path")
    p.add_argument("--th", type=float, default=0.7,
                   help="object-level overlap threshold (fraction of a GT object "
                        "that must be covered)")
    p.add_argument("--b", type=int, default=5,
                   help="buffer in pixels applied to predicted objects when matching")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--k_prevs", type=int, default=0,
                   help="temporal context the split was built with (channel check only)")
    p.add_argument("--attn_unet", action="store_true")
    p.add_argument("--add_attn", action="store_true")
    p.add_argument("--convlstm_unet", action="store_true")
    p.add_argument("--treat_nodata_regions", action="store_true")


def main(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    from ..dataprep.dataset import load_test_dataset
    from ..device import get_device
    from ..models.factory import architecture_from_flags, build_from_checkpoint
    from ..training.evaluate import evaluate

    logging.basicConfig(level=logging.INFO)
    test_data = load_test_dataset(args.test_data_path)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    logging.info(f"{len(test_data)} test patches from {args.test_data_path}")

    device = get_device()
    state_dict = torch.load(args.model, map_location=device)
    loaded = build_from_checkpoint(
        state_dict,
        arch=architecture_from_flags(
            convlstm_unet=args.convlstm_unet, attn_unet=args.attn_unet, add_attn=args.add_attn
        ),
        n_channels=args.k_prevs + 1,
        n_classes=1,
        bilinear=False,
        treat_nodata_regions=args.treat_nodata_regions,
    )
    net = loaded.model
    net.to(device=device)
    net.load_state_dict(state_dict)
    net.eval()
    logging.info(f"architecture {loaded.architecture} ({loaded.n_channels} in-channels) on {device}")

    dice = evaluate(net, test_loader, device, amp=False, mode="test",
                    th=args.th, buffer=args.b)
    logging.info(f"test dice score: {dice}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    main(parser.parse_args())
