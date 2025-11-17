# -*- coding: utf-8 -*-
"""
이미지 + 라벨맵 동시 증강 (회전, 쉬프트, 플립 등)

INPUT:
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/images       (원본 이미지, .jpg)
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/labelmaps    (RGB labelmap, .png)

OUTPUT:
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/aug_images
  /home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/aug_labelmaps

증강 종류:
- 좌우 뒤집기
- 상하 뒤집기
- 90°, 180°, 270° 회전
- 랜덤 회전 (-30 ~ +30도)
- 랜덤 쉬프트(가로/세로 각 10% 범위)
"""

import os
from glob import glob
import random
from PIL import Image, ImageOps, ImageChops

# ----------------
IMG_DIR = "/home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/TrainDataSet_Oct_50m/Images"
MASK_DIR = "/home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/TrainDataSet_Oct_50m/SegmentationClass"
OUT_IMG_DIR = "/home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/TrainDataSet_Oct_50m/Aug_Images"
OUT_MASK_DIR = "/home/keti_taehoon/SHRC_DeepLab3Plus_Learning/Data/TrainDataSet_Oct_50m/Aug_SegmentationClass"
# ----------------


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


ensure_dir(OUT_IMG_DIR)
ensure_dir(OUT_MASK_DIR)


def random_shift(img, mask, max_shift_ratio=0.1):
    """이미지와 마스크를 동일하게 랜덤 쉬프트"""
    w, h = img.size
    max_dx = int(w * max_shift_ratio)
    max_dy = int(h * max_shift_ratio)

    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    img_shifted = ImageChops.offset(img, dx, dy)
    mask_shifted = ImageChops.offset(mask, dx, dy)

    return img_shifted, mask_shifted


def random_rotation(img, mask, max_angle=30):
    """랜덤 회전 (-max_angle ~ +max_angle)"""
    angle = random.uniform(-max_angle, max_angle)

    img_rot = img.rotate(angle, resample=Image.BILINEAR)
    mask_rot = mask.rotate(angle, resample=Image.NEAREST)

    return img_rot, mask_rot


def save_pair(stem, idx, img, mask):
    """저장 함수"""
    img.save(os.path.join(OUT_IMG_DIR, f"{stem}_aug_{idx:03d}.jpg"))
    mask.save(os.path.join(OUT_MASK_DIR, f"{stem}_aug_{idx:03d}.png"))


def augment(stem, img_path, mask_path):
    # 이미 증강된 결과가 존재하는지 검사
    pattern = os.path.join(OUT_IMG_DIR, f"{stem}_aug_*.jpg")
    existing = glob(pattern)

    if len(existing) > 0:
        print(f"[SKIP] {stem} 이미 증강됨 → 건너뜀")
        return
    
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    idx = 1

    # 좌우 플립
    img2 = ImageOps.mirror(img)
    mask2 = ImageOps.mirror(mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 상하 플립
    img2 = ImageOps.flip(img)
    mask2 = ImageOps.flip(mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 90도 회전
    img2 = img.rotate(90, expand=True)
    mask2 = mask.rotate(90, expand=True, resample=Image.NEAREST)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 180도 회전
    img2 = img.rotate(180, expand=True)
    mask2 = mask.rotate(180, expand=True, resample=Image.NEAREST)
    save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 회전
    # img2, mask2 = random_rotation(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # # 랜덤 쉬프트
    # img2, mask2 = random_shift(img, mask)
    # save_pair(stem, idx, img2, mask2); idx += 1

    # 270도 회전
    img2 = img.rotate(270, expand=True)
    mask2 = mask.rotate(270, expand=True, resample=Image.NEAREST)
    save_pair(stem, idx, img2, mask2); idx += 1

    # 랜덤 회전
    img2, mask2 = random_rotation(img, mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    # 랜덤 쉬프트
    img2, mask2 = random_shift(img, mask)
    save_pair(stem, idx, img2, mask2); idx += 1

    print(f"[OK] Augmented {stem}")


def main():
    image_files = []
    exts = ["*.png", "*.jpg"]
    for ext in exts:
        pattern = os.path.join(IMG_DIR, ext)
        image_files.extend(glob(pattern))

    for img_path in image_files:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, stem + ".png")

        if not os.path.isfile(mask_path):
            print(f"[WARN] Missing mask for {stem}, skipped")
            continue

        augment(stem, img_path, mask_path)

    print("Augmentation complete!")


if __name__ == "__main__":
    main()
