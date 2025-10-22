"""
Create a smaller dataset that fits in Google Drive (15GB limit)
Samples 20% of each class while maintaining balance
"""

import shutil
from pathlib import Path
import random
from tqdm import tqdm

def create_reduced_dataset(
    source_dir='Behaviors_Features',
    output_dir='Behaviors_Features_Small',
    sample_percent=0.20  # Use 20% of data
):
    """
    Create smaller dataset by sampling images
    27GB -> ~5-6GB
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Remove output if exists
    if output_path.exists():
        shutil.rmtree(output_path)
    
    print(f"Creating reduced dataset: {sample_percent*100}% of original")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()
    
    total_original = 0
    total_sampled = 0
    
    # Process each behavior class
    behavior_classes = [
        'Looking_Forward', 'Raising_Hand', 'Reading', 'Sleeping',
        'Standing', 'Turning_Around', 'Writting'
    ]
    
    for behavior in behavior_classes:
        behavior_source = source_path / behavior
        
        if not behavior_source.exists():
            print(f"Skipping {behavior} - not found")
            continue
        
        print(f"\nProcessing {behavior}...")
        
        # Process each student ID
        for student_id in ['ID1', 'ID2', 'ID3', 'ID4']:
            student_source = behavior_source / student_id
            
            if not student_source.exists():
                continue
            
            # Get all PNG files recursively
            all_images = list(student_source.rglob('*.png'))
            total_original += len(all_images)
            
            # Sample images
            n_sample = int(len(all_images) * sample_percent)
            sampled_images = random.sample(all_images, n_sample)
            total_sampled += len(sampled_images)
            
            print(f"  {student_id}: {len(all_images):,} -> {len(sampled_images):,} images")
            
            # Copy sampled images
            for img_path in sampled_images:
                # Maintain directory structure
                relative_path = img_path.relative_to(source_path)
                output_img_path = output_path / relative_path
                
                # Create parent directories
                output_img_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(img_path, output_img_path)
    
    print("\n" + "="*70)
    print("DATASET REDUCTION COMPLETE")
    print("="*70)
    print(f"Original images: {total_original:,}")
    print(f"Sampled images: {total_sampled:,}")
    print(f"Reduction: {(1 - total_sampled/total_original)*100:.1f}%")
    print(f"\nNew dataset location: {output_dir}/")
    print(f"\nNow compress this smaller dataset:")
    print(f"  Compress-Archive -Path {output_dir} -DestinationPath {output_dir}.zip")
    print("="*70)


if __name__ == '__main__':
    import sys
    
    print("="*70)
    print("DATASET REDUCTION TOOL")
    print("="*70)
    print("\nThis will create a smaller dataset that fits in Google Drive")
    print("Original: ~27 GB")
    print("Reduced (20%): ~5-6 GB")
    print()
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    create_reduced_dataset(
        source_dir='Behaviors_Features',
        output_dir='Behaviors_Features_Small',
        sample_percent=0.20  # 20% of data = ~50K images
    )
    
    print("\n\nNext step:")
    print("Run this to compress:")
    print("  Compress-Archive -Path Behaviors_Features_Small -DestinationPath Behaviors_Features_Small.zip")

