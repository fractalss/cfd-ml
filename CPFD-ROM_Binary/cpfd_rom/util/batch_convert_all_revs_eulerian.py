import os
import glob
import subprocess

# --- CONFIGURATION ---
BASE_DIR = '/data3/sauravmitra/ML_CFD_PINN/Gasifier'  # Update this to your base folder
SCRIPT_NAME = 'convert_cells_to_npy.py'              # Make sure this is the correct filename

# --- MAIN ---
def main():
    rev_dirs = sorted(glob.glob(os.path.join(BASE_DIR, 'Rev6')))

    if not rev_dirs:
        print(f"[ERROR] No Rev* folders found under {BASE_DIR}")
        return

    print(f"[INFO] Found {len(rev_dirs)} Rev folders: {[os.path.basename(d) for d in rev_dirs]}")

    for rev_dir in rev_dirs:
        rev_name = os.path.basename(rev_dir)
        output_dir = os.path.join(BASE_DIR, f"{rev_name}_npy")
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            'python',
            SCRIPT_NAME,
            '--input_dir', rev_dir,
            '--output_dir', output_dir
        ]

        print(f"[INFO] Converting {rev_name} -> {output_dir}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Conversion failed for {rev_name}: {e}")

    print("[INFO] Batch conversion complete for all Rev folders.")

if __name__ == "__main__":
    main()
