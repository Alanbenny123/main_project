"""
Dataset Analyzer for Behavior and Face Recognition Datasets
Analyzes Behaviors_Features/ and NDB/ directories
"""

import os
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Tuple


class DatasetAnalyzer:
    """Analyze both behavior and face datasets"""
    
    def __init__(self, behavior_dir='Behaviors_Features', face_dir='NDB'):
        self.behavior_dir = Path(behavior_dir)
        self.face_dir = Path(face_dir)
    
    
    def analyze_behaviors(self) -> Dict:
        """Analyze behavior classification dataset"""
        print("="*70)
        print("BEHAVIOR DATASET ANALYSIS")
        print("="*70)
        
        stats = {
            'classes': {},
            'total_images': 0,
            'total_videos': 0,
            'images_per_class': {},
            'images_per_id': defaultdict(int),
            'class_balance': {}
        }
        
        # Behavior classes
        behavior_classes = [
            'Looking_Forward',
            'Raising_Hand', 
            'Reading',
            'Sleeping',
            'Standing',
            'Turning_Around',
            'Writting'
        ]
        
        for behavior in behavior_classes:
            behavior_path = self.behavior_dir / behavior
            
            if not behavior_path.exists():
                print(f"⚠ Warning: {behavior} directory not found")
                continue
            
            class_stats = {
                'total_images': 0,
                'ids': {}
            }
            
            # Analyze each student ID
            for student_id in ['ID1', 'ID2', 'ID3', 'ID4']:
                id_path = behavior_path / student_id
                
                if not id_path.exists():
                    continue
                
                # Count PNG images in all subdirectories
                png_count = len(list(id_path.rglob('*.png')))
                
                class_stats['ids'][student_id] = png_count
                class_stats['total_images'] += png_count
                stats['images_per_id'][student_id] += png_count
            
            stats['classes'][behavior] = class_stats
            stats['images_per_class'][behavior] = class_stats['total_images']
            stats['total_images'] += class_stats['total_images']
        
        # Calculate class balance
        if stats['total_images'] > 0:
            for behavior, count in stats['images_per_class'].items():
                percentage = (count / stats['total_images']) * 100
                stats['class_balance'][behavior] = f"{percentage:.2f}%"
        
        # Print analysis
        print(f"\nTotal Images: {stats['total_images']:,}")
        print(f"Classes: {len(behavior_classes)}")
        print(f"\n{'Class':<20} {'Images':<12} {'Balance':<10} {'ID1':<8} {'ID2':<8} {'ID3':<8} {'ID4':<8}")
        print("-"*70)
        
        for behavior in behavior_classes:
            if behavior in stats['classes']:
                cls = stats['classes'][behavior]
                total = cls['total_images']
                balance = stats['class_balance'].get(behavior, '0%')
                id1 = cls['ids'].get('ID1', 0)
                id2 = cls['ids'].get('ID2', 0)
                id3 = cls['ids'].get('ID3', 0)
                id4 = cls['ids'].get('ID4', 0)
                
                print(f"{behavior:<20} {total:<12,} {balance:<10} {id1:<8,} {id2:<8,} {id3:<8,} {id4:<8,}")
        
        print("-"*70)
        print(f"\n{'Student ID':<12} {'Total Images':<12}")
        for sid in ['ID1', 'ID2', 'ID3', 'ID4']:
            print(f"{sid:<12} {stats['images_per_id'][sid]:,}")
        
        return stats
    
    
    def analyze_faces(self) -> Dict:
        """Analyze face recognition dataset (NDB)"""
        print("\n" + "="*70)
        print("FACE RECOGNITION DATASET ANALYSIS")
        print("="*70)
        
        stats = {
            'total_students': 0,
            'total_images': 0,
            'students': {},
            'images_per_student': [],
            'min_images': float('inf'),
            'max_images': 0,
            'avg_images': 0
        }
        
        # Scan NDB directory
        if not self.face_dir.exists():
            print(f"⚠ Warning: {self.face_dir} not found")
            return stats
        
        student_dirs = sorted([d for d in self.face_dir.iterdir() if d.is_dir()])
        
        for student_dir in student_dirs:
            # Count JPG images
            jpg_count = len(list(student_dir.glob('*.jpg')))
            
            # Parse student info from folder name (e.g., ASI22CA009_Alan_Benny)
            folder_name = student_dir.name
            parts = folder_name.split('_')
            student_id = parts[0] if parts else folder_name
            student_name = '_'.join(parts[1:]) if len(parts) > 1 else 'Unknown'
            
            stats['students'][folder_name] = {
                'id': student_id,
                'name': student_name,
                'images': jpg_count,
                'path': str(student_dir)
            }
            
            stats['total_images'] += jpg_count
            stats['images_per_student'].append(jpg_count)
            stats['min_images'] = min(stats['min_images'], jpg_count)
            stats['max_images'] = max(stats['max_images'], jpg_count)
        
        stats['total_students'] = len(stats['students'])
        
        if stats['total_students'] > 0:
            stats['avg_images'] = stats['total_images'] / stats['total_students']
        
        # Print analysis
        print(f"\nTotal Students: {stats['total_students']}")
        print(f"Total Images: {stats['total_images']:,}")
        print(f"Images per student: Min={stats['min_images']}, Max={stats['max_images']}, Avg={stats['avg_images']:.1f}")
        
        print(f"\n{'Student ID':<15} {'Name':<30} {'Images':<8}")
        print("-"*70)
        
        for folder_name, info in sorted(stats['students'].items())[:10]:  # Show first 10
            print(f"{info['id']:<15} {info['name']:<30} {info['images']:<8}")
        
        if stats['total_students'] > 10:
            print(f"... and {stats['total_students'] - 10} more students")
        
        return stats
    
    
    def check_data_quality(self) -> Dict:
        """Check for potential data quality issues"""
        print("\n" + "="*70)
        print("DATA QUALITY CHECK")
        print("="*70)
        
        issues = {
            'behavior': [],
            'face': []
        }
        
        # Check behavior dataset
        behavior_stats = self.analyze_behaviors()
        
        # Check for class imbalance
        total = behavior_stats['total_images']
        for behavior, count in behavior_stats['images_per_class'].items():
            percentage = (count / total) * 100
            if percentage < 5:
                issues['behavior'].append(f"[!] {behavior}: Very low representation ({percentage:.1f}%)")
            elif percentage > 40:
                issues['behavior'].append(f"[!] {behavior}: Dominant class ({percentage:.1f}%)")
        
        # Check face dataset
        face_stats = self.analyze_faces()
        
        # Check for students with too few images
        for student, info in face_stats['students'].items():
            if info['images'] < 5:
                issues['face'].append(f"[!] {info['id']}: Only {info['images']} images (need >=5 for good embeddings)")
        
        # Print issues
        print("\nIssues Found:")
        
        if issues['behavior']:
            print("\nBehavior Dataset:")
            for issue in issues['behavior']:
                print(f"  {issue}")
        else:
            print("\n[OK] Behavior dataset looks good")
        
        if issues['face']:
            print("\nFace Dataset:")
            for issue in issues['face'][:10]:  # Show first 10
                print(f"  {issue}")
            if len(issues['face']) > 10:
                print(f"  ... and {len(issues['face']) - 10} more")
        else:
            print("\n[OK] Face dataset looks good")
        
        return issues
    
    
    def generate_split_recommendations(self) -> Dict:
        """Recommend train/val/test splits"""
        print("\n" + "="*70)
        print("DATASET SPLIT RECOMMENDATIONS")
        print("="*70)
        
        recommendations = {}
        
        # Behavior dataset: Use ID-based split to prevent data leakage
        print("\nBehavior Classification:")
        print("  Strategy: ID-based split (prevent same person in train/test)")
        print("  Recommendation:")
        print("    - Train: ID1, ID2, ID3 (75%)")
        print("    - Validation: ID4 (12.5%)")  
        print("    - Test: ID4 (12.5%) or use cross-validation")
        print("  Alternative: K-fold cross-validation across IDs")
        
        recommendations['behavior'] = {
            'strategy': 'id_based',
            'train_ids': ['ID1', 'ID2', 'ID3'],
            'val_ids': ['ID4'],
            'test_ids': ['ID4']
        }
        
        # Face dataset: Random split or student-based
        print("\nFace Recognition:")
        print("  Strategy: Per-student split (some images for train, some for test)")
        print("  Recommendation:")
        print("    - Train: 70% of each student's images")
        print("    - Validation: 15% of each student's images")
        print("    - Test: 15% of each student's images")
        print("  Note: Ensures all students in all splits for proper evaluation")
        
        recommendations['face'] = {
            'strategy': 'per_student_split',
            'train_ratio': 0.70,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        }
        
        return recommendations
    
    
    def save_analysis(self, output_file='dataset_analysis.json'):
        """Save full analysis to JSON"""
        analysis = {
            'behavior': self.analyze_behaviors(),
            'face': self.analyze_faces(),
            'recommendations': self.generate_split_recommendations()
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\nAnalysis saved to {output_file}")
        return analysis


if __name__ == '__main__':
    analyzer = DatasetAnalyzer()
    
    # Run full analysis
    behavior_stats = analyzer.analyze_behaviors()
    face_stats = analyzer.analyze_faces()
    
    # Quality check
    issues = analyzer.check_data_quality()
    
    # Split recommendations
    recommendations = analyzer.generate_split_recommendations()
    
    # Save results
    analyzer.save_analysis()
    
    print("\n" + "="*70)
    print("[DONE] Analysis Complete!")
    print("="*70)

