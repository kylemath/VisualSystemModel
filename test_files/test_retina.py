"""
Test script for the retinal layer.
Demonstrates the photoreceptor functionality.
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from backup_files.retina import RetinaLayer
import matplotlib.pyplot as plt

def test_basic_functionality():
    """Test basic retinal layer functionality."""
    print("Testing Retinal Layer...")
    print("=" * 60)
    
    # Create a retinal layer
    grid_size = 32
    retina = RetinaLayer(grid_size=grid_size)
    
    print(f"✓ Created retinal layer with {grid_size}×{grid_size} grid")
    print(f"✓ Total photoreceptors: {retina.total_photoreceptors:,}")
    print(f"  - Rods: {len(retina.photoreceptors['left']['rods']):,} per eye")
    print(f"  - Red cones: {len(retina.photoreceptors['left']['red_cones']):,} per eye")
    print(f"  - Green cones: {len(retina.photoreceptors['left']['green_cones']):,} per eye")
    print(f"  - Blue cones: {len(retina.photoreceptors['left']['blue_cones']):,} per eye")
    
    # Create a test image (vertical color bands)
    test_image = np.zeros((grid_size, grid_size, 3))
    band_width = grid_size // 3
    test_image[:, :band_width, 0] = 1.0  # Red band
    test_image[:, band_width:2*band_width, 1] = 1.0  # Green band
    test_image[:, 2*band_width:, 2] = 1.0  # Blue band
    
    print("\n✓ Created test pattern (RGB color bands)")
    
    # Process through retina
    retina.process_image(test_image, test_image, intensity=1.0)
    print("✓ Processed image through retinal layer")
    
    # Get activity maps
    left_maps = retina.get_activity_maps('left')
    
    print("\n" + "=" * 60)
    print("Photoreceptor Activity Analysis:")
    print("=" * 60)
    
    for receptor_type, activity_map in left_maps.items():
        mean_activity = np.mean(activity_map)
        max_activity = np.max(activity_map)
        active_count = np.sum(activity_map > 0.1)
        
        print(f"\n{receptor_type.upper()}:")
        print(f"  Mean activity: {mean_activity:.3f}")
        print(f"  Max activity: {max_activity:.3f}")
        print(f"  Active receptors: {active_count}/{grid_size*grid_size}")
    
    # Get summary
    summary = retina.get_summary()
    print("\n" + "=" * 60)
    print("System Summary:")
    print("=" * 60)
    print(f"Grid size: {summary['grid_size']}×{summary['grid_size']}")
    print(f"Total photoreceptors: {summary['total_photoreceptors']:,}")
    
    print("\nLeft Eye Stats:")
    for receptor_type, stats in summary['left_eye'].items():
        print(f"  {receptor_type}: {stats['mean_activity']:.2%} mean activity")
    
    print("\n✓ All tests passed!")
    return retina, left_maps


def visualize_with_matplotlib(retina, maps):
    """Visualize the photoreceptor activity using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for macOS
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Retinal Photoreceptor Activity Maps', fontsize=16)
        
        receptor_types = ['rods', 'red_cones', 'green_cones', 'blue_cones']
        titles = ['Rods (Low-light)', 'Red Cones (L)', 'Green Cones (M)', 'Blue Cones (S)']
        cmaps = ['gray', 'Reds', 'Greens', 'Blues']
        
        for idx, (receptor_type, title, cmap) in enumerate(zip(receptor_types, titles, cmaps)):
            ax = axes[idx // 2, idx % 2]
            im = ax.imshow(maps[receptor_type], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('retina_activity.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved visualization to 'retina_activity.png'")
        print("  (Matplotlib visualization saved)")
        
    except Exception as e:
        print(f"\nNote: Could not create matplotlib visualization: {e}")
        print("  The web interface provides full visualization capabilities.")


if __name__ == "__main__":
    print("\n" + "🧠 BIO-INSPIRED NEURAL NETWORK - RETINAL LAYER TEST ".center(60, "="))
    print()
    
    retina, maps = test_basic_functionality()
    
    # Try to visualize (optional)
    try:
        visualize_with_matplotlib(retina, maps)
    except:
        pass
    
    print("\n" + "=" * 60)
    print("\n🌐 For interactive visualization, run:")
    print("   python server.py")
    print("   Then open http://localhost:5000 in your browser")
    print("\n" + "=" * 60 + "\n")

