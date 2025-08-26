#!/usr/bin/env python3
"""
Convert COCO format annotations to YOLO format
"""
import json
import os
import shutil
from pathlib import Path

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """
    Convert COCO bbox format [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All values normalized to 0-1
    """
    x, y, w, h = coco_bbox
    
    # Convert to YOLO format (center coordinates, normalized)
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return [x_center, y_center, width, height]

def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, split_name):
    """
    Convert COCO format to YOLO format
    """
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_images_dir = Path(output_dir) / 'images' / split_name
    output_labels_dir = Path(output_dir) / 'labels' / split_name
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create category mapping (COCO ID -> YOLO class index)
    # COCO IDs: [1, 2, 3, 4, 6, 8] -> YOLO classes: [0, 1, 2, 3, 4, 5]
    coco_to_yolo_map = {}
    for idx, category in enumerate(coco_data['categories']):
        coco_to_yolo_map[category['id']] = idx
    
    print(f"Category mapping for {split_name}:")
    for category in coco_data['categories']:
        yolo_idx = coco_to_yolo_map[category['id']]
        print(f"  COCO ID {category['id']} ({category['name']}) -> YOLO class {yolo_idx}")
    
    # Create image lookup
    image_lookup = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Process each image
    processed_count = 0
    for img_id, img_info in image_lookup.items():
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Copy image file
        src_img_path = Path(images_dir) / img_filename
        dst_img_path = output_images_dir / img_filename
        
        if src_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
            
            # Create YOLO label file
            label_filename = img_filename.replace('.jpg', '.txt')
            label_path = output_labels_dir / label_filename
            
            # Get annotations for this image
            img_annotations = annotations_by_image.get(img_id, [])
            
            # Write YOLO annotations
            with open(label_path, 'w') as f:
                for ann in img_annotations:
                    # Convert category ID to YOLO class index
                    yolo_class = coco_to_yolo_map[ann['category_id']]
                    
                    # Convert bbox to YOLO format
                    yolo_bbox = coco_to_yolo_bbox(ann['bbox'], img_width, img_height)
                    
                    # Write to file
                    f.write(f"{yolo_class} {' '.join(map(str, yolo_bbox))}\n")
            
            processed_count += 1
        else:
            print(f"Warning: Image file not found: {src_img_path}")
    
    print(f"Processed {processed_count} images for {split_name}")
    return processed_count

def create_yaml_config(output_dir, categories):
    """
    Create YOLO dataset configuration YAML file
    """
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Dataset path (relative to this file)
path: {output_dir}

# Train and validation sets
train: images/train
val: images/val

# Number of classes
nc: {len(categories)}

# Class names
names:
"""
    
    # Add class names in order
    for idx, category in enumerate(categories):
        yaml_content += f"  {idx}: {category['name']}\n"
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created YAML configuration: {yaml_path}")
    return yaml_path

def main():
    # Paths
    base_dir = Path("d:/wis coco/coco-datasets")
    coco_dir = base_dir / "filtered_12k_coco"
    output_dir = Path("D:/wis Thesis/Yolov8-Thesis/datasets/coco_vehicle_person")
    
    # Input paths
    train_json = coco_dir / "instances_train.json"
    val_json = coco_dir / "instances_val.json"
    train_images = coco_dir / "train" / "images"
    val_images = coco_dir / "val" / "images"
    
    print("Starting COCO to YOLO conversion...")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert training set
    print("\n=== Converting Training Set ===")
    train_count = convert_coco_to_yolo(train_json, train_images, output_dir, 'train')
    
    # Convert validation set
    print("\n=== Converting Validation Set ===")
    val_count = convert_coco_to_yolo(val_json, val_images, output_dir, 'val')
    
    # Create YAML configuration
    print("\n=== Creating YAML Configuration ===")
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    
    yaml_path = create_yaml_config(output_dir, train_data['categories'])
    
    print(f"\n=== Conversion Complete ===")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Total images: {train_count + val_count}")
    print(f"Dataset location: {output_dir}")
    print(f"YAML config: {yaml_path}")
    
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── dataset.yaml")
    print(f"    ├── images/")
    print(f"    │   ├── train/ ({train_count} images)")
    print(f"    │   └── val/ ({val_count} images)")
    print(f"    └── labels/")
    print(f"        ├── train/ ({train_count} .txt files)")
    print(f"        └── val/ ({val_count} .txt files)")

if __name__ == "__main__":
    main()
