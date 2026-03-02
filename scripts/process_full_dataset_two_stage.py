#!/usr/bin/env python3
"""
Process Full Dataset with Two-Stage Detection
==============================================
Apply two-stage YOLO detection to all images in train/valid/test splits.

This script:
1. Loads all images from each split
2. Runs two-stage detection (card → fields)
3. Saves cropped fields to rec/images/two_stage/
4. Creates metadata CSV for OCR labeling

Usage:
    python scripts/process_full_dataset_two_stage.py [--splits train valid test] [--limit N]
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import pandas as pd
from tqdm import tqdm
from src.card_detector import CardDetector, load_card_detector


def get_image_files(split_path: Path) -> list:
    """Get all image files from a split directory."""
    # Check if images are in subdirectory
    images_dir = split_path / "images"
    if images_dir.exists():
        split_path = images_dir
    
    return sorted(
        list(split_path.glob("*.jpg")) +
        list(split_path.glob("*.png")) +
        list(split_path.glob("*.jpeg"))
    )


def process_split(
    split: str,
    split_path: Path,
    detector: CardDetector,
    output_dir: Path,
    limit: int = None,
    translate_to_project: bool = True,
) -> tuple:
    """
    Process all images in a single split.
    
    Args:
        split: Split name (train/valid/test)
        split_path: Path to split directory
        detector: CardDetector instance
        output_dir: Directory to save cropped fields
        limit: Maximum number of images to process (None = all)
        translate_to_project: Translate NASO7Y classes to project format
        
    Returns:
        Tuple of (metadata_list, stats_dict)
    """
    image_files = get_image_files(split_path)
    
    if limit:
        image_files = image_files[:limit]
    
    if not image_files:
        print(f"⚠️  No images found in {split_path}")
        return [], {}
    
    print(f"\n📋 Processing split: {split}")
    print(f"   Images: {len(image_files)}")
    print(f"   Output: {output_dir}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    stats = {
        'processed': 0,
        'success': 0,
        'errors': 0,
        'total_crops': 0,
        'fields_detected': {},
    }
    
    pbar = tqdm(image_files, desc=f"[{split}]")
    
    for img_path in pbar:
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                stats['errors'] += 1
                continue
            
            # Run two-stage detection
            card_crop, fields = detector.detect_full(
                image,
                translate_to_project=translate_to_project,
            )
            
            if not fields:
                stats['processed'] += 1
                continue
            
            # Save crops and collect metadata
            for field_name, (crop, conf) in fields.items():
                # Generate filename
                save_name = f"{split}_{img_path.stem}_{field_name}.jpg"
                save_path = output_dir / save_name
                
                # Save crop
                cv2.imwrite(str(save_path), crop)
                
                # Add to metadata
                metadata.append({
                    'image_path': f"rec/images/two_stage/{save_name}",
                    'field': field_name,
                    'class_id': -1,  # Not applicable for NASO7Y
                    'split': split,
                    'orig_image': img_path.name,
                    'confidence': round(conf, 3),
                    'label_text': '',  # To be filled by OCR
                    'card_cropped': True,
                    'processed_at': datetime.now().isoformat(),
                })
                
                # Update stats
                stats['total_crops'] += 1
                stats['fields_detected'][field_name] = stats['fields_detected'].get(field_name, 0) + 1
            
            stats['success'] += 1
            stats['processed'] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'success': stats['success'],
                'crops': stats['total_crops'],
            })
            
        except Exception as e:
            stats['errors'] += 1
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
    
    return metadata, stats


def main():
    parser = argparse.ArgumentParser(
        description="Process full dataset with two-stage detection"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Splits to process (default: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images per split (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rec/images/two_stage",
        help="Output directory for cropped fields"
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="crops_metadata_two_stage.csv",
        help="Output metadata CSV filename"
    )
    parser.add_argument(
        "--translate-to-project",
        action="store_true",
        default=True,
        help="Translate NASO7Y classes to project format (default: True)"
    )
    parser.add_argument(
        "--no-translate",
        action="store_false",
        dest="translate_to_project",
        help="Keep NASO7Y class names"
    )
    parser.add_argument(
        "--card-model",
        type=str,
        default="weights/card_detection.pt",
        help="Path to card detection model"
    )
    parser.add_argument(
        "--field-model",
        type=str,
        default="weights/field_detection.pt",
        help="Path to field detection model"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("🚀 TWO-STAGE DETECTION - FULL DATASET PROCESSING")
    print("=" * 60)
    print()
    print("📋 Configuration:")
    print(f"   Splits: {args.splits}")
    print(f"   Limit per split: {args.limit or 'all'}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Metadata file: {args.metadata_file}")
    print(f"   Translate to project format: {args.translate_to_project}")
    print(f"   Card model: {args.card_model}")
    print(f"   Field model: {args.field_model}")
    print()
    
    # Check if models exist
    card_model_path = ROOT / args.card_model
    field_model_path = ROOT / args.field_model
    
    if not card_model_path.exists():
        print(f"❌ Card model not found: {card_model_path}")
        print("   Run: python scripts/download_weights.py")
        return
    
    if not field_model_path.exists():
        print(f"❌ Field model not found: {field_model_path}")
        print("   Run: python scripts/download_weights.py")
        return
    
    # Load detector
    print("⏳ Loading detection models...")
    detector = load_card_detector(
        card_model_path=str(card_model_path),
        field_model_path=str(field_model_path),
    )
    print("✅ Models loaded successfully!")
    print()
    
    # Process each split
    all_metadata = []
    all_stats = {}
    
    start_time = time.time()
    
    for split in args.splits:
        split_path = ROOT / split
        
        if not split_path.exists():
            print(f"⚠️  Split directory not found: {split_path}")
            continue
        
        output_dir = ROOT / args.output_dir
        
        metadata, stats = process_split(
            split=split,
            split_path=split_path,
            detector=detector,
            output_dir=output_dir,
            limit=args.limit,
            translate_to_project=args.translate_to_project,
        )
        
        all_metadata.extend(metadata)
        all_stats[split] = stats
    
    elapsed_time = time.time() - start_time
    
    # Save metadata
    metadata_path = None
    if all_metadata:
        df_metadata = pd.DataFrame(all_metadata)
        metadata_path = ROOT / args.metadata_file
        df_metadata.to_csv(metadata_path, index=False, encoding='utf-8-sig')
        print()
        print(f"✅ Metadata saved to: {metadata_path}")
        print(f"   Total records: {len(df_metadata)}")
    else:
        print()
        print("⚠️  No metadata to save")
    
    # Print summary
    print()
    print("=" * 60)
    print("📊 PROCESSING SUMMARY")
    print("=" * 60)
    print()
    
    total_processed = 0
    total_success = 0
    total_errors = 0
    total_crops = 0
    
    for split, stats in all_stats.items():
        print(f"📋 Split: {split}")
        print(f"   Processed: {stats.get('processed', 0)}")
        print(f"   Success: {stats.get('success', 0)}")
        print(f"   Errors: {stats.get('errors', 0)}")
        print(f"   Total crops: {stats.get('total_crops', 0)}")
        
        if stats.get('fields_detected'):
            print(f"   Fields detected:")
            for field, count in sorted(stats['fields_detected'].items()):
                print(f"      - {field:20s}: {count}")
        
        print()
        
        total_processed += stats.get('processed', 0)
        total_success += stats.get('success', 0)
        total_errors += stats.get('errors', 0)
        total_crops += stats.get('total_crops', 0)
    
    print("-" * 60)
    print(f"📈 TOTALS:")
    print(f"   Images processed: {total_processed}")
    print(f"   Successful: {total_success}")
    print(f"   Errors: {total_errors}")
    print(f"   Total field crops: {total_crops}")
    print(f"   Time elapsed: {elapsed_time:.1f}s")
    print(f"   Avg per image: {elapsed_time / max(total_processed, 1):.2f}s")
    print()
    print("📂 Output:")
    print(f"   Crops: {ROOT / args.output_dir}")
    print(f"   Metadata: {metadata_path}")
    print()
    print("🚀 Next Steps:")
    print("   1. Review metadata: head crops_metadata_two_stage.csv")
    print("   2. Label with OCR: python scripts/label_crops.py --method qari")
    print("   3. Train PaddleOCR: See notebook 02_label_and_train.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
