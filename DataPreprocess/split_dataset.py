import argparse
import random
import shutil
from pathlib import Path

# ------------ Config ------------
DEFAULT_CATEGORIES = ["normal", "fog", "rain", "snow", "fgsm_attack_dave2v1", "fgsm_attack_epoch","pgd_attack_dave2v1","pgd_attack_epoch","sp_attack_epoch","spsa_attack_epoch"]
ALLOWED_NORMAL_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # normal images
ALLOWED_GEN_EXTS = {".png", ".npy", ".jpg"}  # generated images can be png or npy
# --------------------------------

def gather_files(root: Path, category: str, exts: set[str]) -> list[Path]:
    """Recursively gather files inside root/category with given extensions."""
    cat_dir = root / category
    if not cat_dir.exists():
        return []
    files = [p for p in cat_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return files

def relative_in_category(file_path: Path, category_dir: Path) -> Path:
    """Return file_path relative to the category directory."""
    return file_path.relative_to(category_dir)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def find_generated_for_id(
    dataset_root: Path,
    categories: list[str],
    normal_rel_path: Path,
    normal_stem: str,
) -> list[tuple[str, Path]]:
    """
    For each non-normal category, try to find files with the same stem at the same relative folder.
    Return list of tuples (category, file_path).
    """
    found = []
    for cat in categories:
        if cat == "normal":
            continue
        # Look in the *same relative directory* under this category
        gen_dir = dataset_root / cat / normal_rel_path.parent
        if not gen_dir.exists():
            continue

        # Match same stem with any allowed generated extension
        for ext in ALLOWED_GEN_EXTS:
            candidate = gen_dir / f"{normal_stem}{ext}"
            if candidate.exists():
                found.append((cat, candidate))

        # If users might have multiple matching files (e.g., multiple variants),
        # also sweep the directory for files that startwith stem (optional).
        # Uncomment if needed:
        # for p in gen_dir.glob(f"{normal_stem}*"):
        #     if p.is_file() and p.suffix.lower() in ALLOWED_GEN_EXTS:
        #         found.append((cat, p))
    return found

def split_train_test(items: list, train_ratio: float, seed: int | None):
    if seed is not None:
        random.seed(seed)
    random.shuffle(items)
    n = len(items)
    train_n = int(n * train_ratio)
    train_items = items[:train_n]
    test_items = items[train_n:]
    return train_items, test_items

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/test preserving category structure and pairing generated files.")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root folder containing category folders (e.g., normal, fog, rain, snow).")
    parser.add_argument("--out_root", type=str, default=None,
                        help="Output root to create train/ and test/ under. Defaults to dataset_root.")
    parser.add_argument("--categories", type=str, nargs="+", default=DEFAULT_CATEGORIES,
                        help=f"Category folder names. Default: {DEFAULT_CATEGORIES}")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="Fraction of normal images to use for train. Default: 0.6 (i.e., 60%).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying. (Default copies)")
    parser.add_argument("--strict_pairs", action="store_true",
                        help="If set, only include a normal image if ALL categories have a corresponding generated file.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else dataset_root
    categories = args.categories
    train_ratio = args.train_ratio
    seed = args.seed
    do_move = args.move
    strict_pairs = args.strict_pairs

    # Validate presence of "normal" category
    if "normal" not in categories:
        raise ValueError("The categories must include 'normal' as the reference folder.")

    normal_dir = dataset_root / "normal"
    if not normal_dir.exists():
        raise FileNotFoundError(f"'normal' directory not found at {normal_dir}")

    # 1) Gather all normal images (recursively)
    normal_files = gather_files(dataset_root, "normal", ALLOWED_NORMAL_EXTS)
    if not normal_files:
        raise RuntimeError("No normal images found.")

    # Map each normal image to: (relative_path_in_normal, stem, full_path)
    normal_records = []
    for nf in normal_files:
        rel = relative_in_category(nf, normal_dir)
        stem = nf.stem  # base filename w/o extension
        normal_records.append((rel, stem, nf))

    # 2) If strict pairing is requested, filter out normals that don't have all generated counterparts
    if strict_pairs:
        filtered = []
        missing_counts = 0
        for rel, stem, nf in normal_records:
            generated = find_generated_for_id(dataset_root, categories, rel, stem)
            # Check that for each non-normal category at least one file exists
            have = {cat for (cat, _) in generated}
            needed = set(categories) - {"normal"}
            if needed.issubset(have):
                filtered.append((rel, stem, nf))
            else:
                missing_counts += 1
        normal_records = filtered
        print(f"[strict_pairs] Excluding {missing_counts} normal images without full generated sets.")
        if not normal_records:
            raise RuntimeError("No normal images left after strict pairing check.")

    # 3) Train/test split based on normal images
    train_normals, test_normals = split_train_test(normal_records, train_ratio, seed)
    print(f"Total normal images: {len(normal_records)}")
    print(f"Train normals: {len(train_normals)}  |  Test normals: {len(test_normals)}")

    # 4) Prepare output dirs (train/test + category subfolders)
    train_root = out_root / "train"
    test_root = out_root / "test"
    for split_root in [train_root, test_root]:
        for cat in categories:
            ensure_dir(split_root / cat)

    # 5) Define copy/move operation
    transfer = shutil.move if do_move else shutil.copy2

    # 6) Function to send a normal image + its generated siblings into the split
    def place_set(split_root: Path, rel: Path, stem: str, normal_full: Path):
        # Normal image destination mirrors same relative path under 'normal'
        dst_normal = split_root / "normal" / rel
        ensure_dir(dst_normal.parent)
        transfer(normal_full, dst_normal)

        # Generated counterparts (if present)
        generated = find_generated_for_id(dataset_root, categories, rel, stem)
        if not generated:
            # It's OK to have no generated files for some items (unless strict_pairs)
            return 0

        copied = 0
        for cat, gpath in generated:
            # Recreate same relative directory and same filename under split_root/cat
            dst_gen = split_root / cat / rel.parent / gpath.name
            ensure_dir(dst_gen.parent)
            transfer(gpath, dst_gen)
            copied += 1
        return copied

    # 7) Do the transfers
    train_gen_copied = 0
    for rel, stem, nf in train_normals:
        train_gen_copied += place_set(train_root, rel, stem, nf)

    test_gen_copied = 0
    for rel, stem, nf in test_normals:
        test_gen_copied += place_set(test_root, rel, stem, nf)

    print("\n=== Summary ===")
    print(f"Train normal files: {len(train_normals)}")
    print(f"Train generated files copied: {train_gen_copied}")
    print(f"Test normal files: {len(test_normals)}")
    print(f"Test generated files copied: {test_gen_copied}")
    if do_move:
        print("Operation: MOVE")
    else:
        print("Operation: COPY")

if __name__ == "__main__":
    main()
