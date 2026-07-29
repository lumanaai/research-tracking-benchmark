#!/usr/bin/env python3
"""Convert a UA-DETRAC sequence into a MOTChallenge-style folder.

Produces a layout that FastTracker's ``tools/track_from_dets.py`` can consume,
using the UA-DETRAC ground-truth boxes as (oracle) detections:

    <out_root>/<SEQ>/
        seqinfo.ini
        gt/gt.txt          # frame,id,left,top,w,h,1,class,1
        img1 -> <DETRAC frames dir>   (symlink, for later visualization)

Example:
    .venv/bin/python scripts/convert/prepare_ua_detrac_mot.py MVI_40213
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

DETRAC_W, DETRAC_H = 960, 540
DETRAC_FPS = 25

VEHICLE_CLASS_ID = {"car": 1, "bus": 2, "van": 3, "others": 4}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UA-DETRAC sequence -> MOT-style folder.")
    p.add_argument("sequence", help="Sequence name, e.g. MVI_40213.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/media/7TBSSD/data/tracking/UA-DETRAC"),
        help="UA-DETRAC dataset root.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("/media/7TBSSD/data/tracking/UA-DETRAC_fasttracker"),
        help="Where to write the MOT-style sequence folder.",
    )
    return p.parse_args()


def resolve_nested(root: Path, name: str) -> Path:
    direct = root / name
    nested = direct / name
    return nested if nested.is_dir() else direct


def find_annotation(dataset_root: Path, seq: str) -> tuple[Path, str]:
    for split in ("DETRAC-Train-Annotations-XML", "DETRAC-Test-Annotations-XML"):
        xml_dir = resolve_nested(dataset_root, split)
        xml_path = xml_dir / f"{seq}.xml"
        if xml_path.is_file():
            kind = "train" if "Train" in split else "test"
            return xml_path, kind
    raise FileNotFoundError(f"No annotation XML found for {seq} under {dataset_root}")


def main() -> None:
    args = parse_args()
    seq = args.sequence
    dataset_root = args.dataset_root.expanduser().resolve()

    xml_path, split = find_annotation(dataset_root, seq)
    images_dir = resolve_nested(dataset_root, "DETRAC-Images") / seq
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing frames for {seq}: {images_dir}")

    root = ET.parse(xml_path).getroot()

    out_seq = (args.out_root.expanduser().resolve()) / seq
    gt_dir = out_seq / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    max_frame = 0
    n_boxes = 0
    with (gt_dir / "gt.txt").open("w") as f:
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

    # Count actual frame images to set seqLength robustly.
    n_images = len(list(images_dir.glob("img*.jpg"))) or len(list(images_dir.glob("*.jpg")))
    seq_length = max(max_frame, n_images)

    seqinfo = out_seq / "seqinfo.ini"
    seqinfo.write_text(
        "[Sequence]\n"
        f"name={seq}\n"
        "imDir=img1\n"
        f"frameRate={DETRAC_FPS}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={DETRAC_W}\n"
        f"imHeight={DETRAC_H}\n"
        "imExt=.jpg\n"
    )

    # Symlink img1 -> DETRAC frames (for later visualization / eval).
    img1 = out_seq / "img1"
    if img1.is_symlink() or img1.exists():
        img1.unlink()
    img1.symlink_to(images_dir)

    print(f"Sequence     : {seq} ({split})")
    print(f"Annotation   : {xml_path}")
    print(f"Frames dir   : {images_dir} ({n_images} images)")
    print(f"Wrote        : {gt_dir / 'gt.txt'} ({n_boxes} boxes, {seq_length} frames)")
    print(f"seqinfo.ini  : {seqinfo}")
    print(f"MOT root     : {args.out_root.expanduser().resolve()}")


if __name__ == "__main__":
    main()
