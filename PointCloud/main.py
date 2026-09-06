# main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

# Set environment variables for compatibility BEFORE importing heavy libraries like Open3D or PyQt
os.environ["OMP_NUM_THREADS"] = '1'         # For OpenMP used by libraries like NumPy/SciPy/Open3D
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # To potentially avoid MKL math lib conflicts

# Import QApplication and style factory
from PyQt5.QtWidgets import QApplication, QStyleFactory

# Import the main GUI class
from gui import PointCloudProcessor

# --- Main Execution ---
if __name__ == '__main__':
    # Print Open3D version for debugging
    try:
        import open3d as o3d
        print(f"Using Open3D version: {o3d.__version__}")
    except ImportError:
        print("Error: Open3D library not found. Please install it (`pip install open3d`)")
        sys.exit(1)
    except Exception as e:
        print(f"Error importing Open3D: {e}")

    print(f"Python version: {sys.version}")

    app = QApplication(sys.argv)

    # Optional: Apply a modern style like Fusion
    if 'Fusion' in QStyleFactory.keys():
        try:
            app.setStyle('Fusion')
            print("Applied Fusion style.")
        except Exception as e_style:
            print(f"Could not apply Fusion style: {e_style}")
    else:
        print(f"Fusion style not available. Available styles: {QStyleFactory.keys()}")

    # Launch main window
    ex = PointCloudProcessor()
    ex.show()

    # Start the event loop
    sys.exit(app.exec_())
