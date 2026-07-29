"""Convert UA-DETRAC XML sequences into MOTChallenge folders."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from mot_pipeline.mot_io import write_seqinfo
from mot_pipeline.paths import MOT_ROOT, RAW_DATASETS

DETRAC_W, DETRAC_H = 960, 540
DETRAC_FPS = 25
VEHICLE_CLASS_ID = {"car": 1, "bus": 2, "van": 3, "others": 4}


def resolve_nested(root: Path, name: str) -> Path:
    direct = root / name
    nested = direct / name
    return nested if nested.is_dir() else direct


def find_annotation(dataset_root: Path, seq: str) -> Tuple[Path, str]:
    for split_name, kind in (
        ("DETRAC-Train-Annotations-XML", "train"),
        ("DETRAC-Test-Annotations-XML", "test"),
    ):
        xml_dir = resolve_nested(dataset_root, split_name)
        xml_path = xml_dir / f"{seq}.xml"
        if xml_path.is_file():
            return xml_path, kind
    raise FileNotFoundError(f"No annotation XML for {seq} under {dataset_root}")


def list_sequences(dataset_root: Path, split: Optional[str] = None) -> List[Tuple[str, str, Path]]:
    """Return [(seq, split, xml_path), ...]."""
    out: List[Tuple[str, str, Path]] = []
    splits = []
    if split in (None, "train", "all"):
        splits.append(("DETRAC-Train-Annotations-XML", "train"))
    if split in (None, "test", "all"):
        splits.append(("DETRAC-Test-Annotations-XML", "test"))
    for split_name, kind in splits:
        xml_dir = resolve_nested(dataset_root, split_name)
        if not xml_dir.is_dir():
            continue
        for xml_path in sorted(xml_dir.glob("*.xml")):
            out.append((xml_path.stem, kind, xml_path))
    return out


def convert_sequence(
    dataset_root: Path,
    seq: str,
    out_root: Path,
    *,
    force: bool = False,
) -> Path:
    xml_path, split = find_annotation(dataset_root, seq)
    images_dir = resolve_nested(dataset_root, "DETRAC-Images") / seq
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing frames for {seq}: {images_dir}")

    out_seq = out_root / split / seq
    gt_path = out_seq / "gt" / "gt.txt"
    if gt_path.is_file() and (out_seq / "seqinfo.ini").is_file() and not force:
        return out_seq

    root = ET.parse(xml_path).getroot()
    gt_dir = out_seq / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    max_frame = 0
    n_boxes = 0
    with gt_path.open("w") as f:
        for frame_elem in root.findall("frame"):
            frame_num = int(frame_elem.get("num"))
            max_frame = max(max_frame, frame_num)
            target_list = frame_elem.find("target_list")
            if target_list is None:
                continue
            for target in target_list.findall("target"):
                tid = int(target.get("id"))
                box = target.find("box")
                left = float(box.get("left"))
                top = float(box.get("top"))
                width = float(box.get("width"))
                height = float(box.get("height"))
                attr = target.find("attribute")
                vtype = attr.get("vehicle_type", "car") if attr is not None else "car"
                cls = VEHICLE_CLASS_ID.get(vtype, 4)
                f.write(
                    f"{frame_num},{tid},{left:.2f},{top:.2f},"
                    f"{width:.2f},{height:.2f},1,{cls},1\n"
                )
                n_boxes += 1

    n_images = len(list(images_dir.glob("img*.jpg"))) or len(
        list(images_dir.glob("*.jpg"))
    )
    seq_length = max(max_frame, n_images)
    write_seqinfo(
        out_seq / "seqinfo.ini",
        name=seq,
        frame_rate=DETRAC_FPS,
        seq_length=seq_length,
        im_width=DETRAC_W,
        im_height=DETRAC_H,
    )

    img1 = out_seq / "img1"
    if img1.is_symlink() or img1.exists():
        if img1.is_dir() and not img1.is_symlink():
            pass
        else:
            img1.unlink()
            img1.symlink_to(images_dir)
    else:
        img1.symlink_to(images_dir)

    _ = n_boxes
    return out_seq


def convert_all(
    dataset_root: Optional[Path] = None,
    out_root: Optional[Path] = None,
    *,
    split: str = "all",
    sequences: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Path:
    dataset_root = Path(dataset_root or RAW_DATASETS["ua_detrac"]).expanduser().resolve()
    out_root = Path(out_root or MOT_ROOT / "UA-DETRAC").expanduser().resolve()
    wanted = set(sequences) if sequences else None
    items = list_sequences(dataset_root, split=None if split == "all" else split)
    for seq, kind, _xml in items:
        if wanted is not None and seq not in wanted:
            continue
        if split not in ("all", kind):
            continue
        convert_sequence(dataset_root, seq, out_root, force=force)
        print(f"  UA-DETRAC {kind}/{seq}")
    return out_root


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="UA-DETRAC → MOT layout under mot/UA-DETRAC/")
    p.add_argument("--dataset-root", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--split", choices=("train", "test", "all"), default="all")
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    out = convert_all(
        args.dataset_root,
        args.out_root,
        split=args.split,
        sequences=args.sequences,
        force=args.force,
    )
    print(f"Done. MOT root: {out}")


if __name__ == "__main__":
    main()
