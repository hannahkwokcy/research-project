import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys
import shutil

# --- 1. HPC CPU CONFIG ---
NUM_CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 32))
tf.config.threading.set_inter_op_parallelism_threads(NUM_CORES)
tf.config.threading.set_intra_op_parallelism_threads(NUM_CORES)

def log(msg):
    print(f"--- [LOG]: {msg}")
    sys.stdout.flush()

# --- 2. PATH CONFIGURATION ---
TRAIN_IMG_DIR = '/home/hlckwok2/research/unet_segmentation/Images'           # Your 200 clean original images
TRAIN_MASK_DIR = '/home/hlckwok2/research/unet_segmentation/Masks'           # Your 200 binary pupil masks
INFERENCE_DIR = '/home/hlckwok2/research/Dataset'    
OUTPUT_DIR    = '/home/hlckwok2/research/segmentation2/Segmented_Results'      # Pass 1 (Strict)
RESCUED_DIR   = '/home/hlckwok2/research/segmentation2/Rescued_Results'        # Pass 2 (Relaxed)
ULTRA_DIR     = '/home/hlckwok2/research/segmentation2/Ultra_Rescued_Results'    # Pass 3 (Extreme)
SKIPPED_DIR   = '/home/hlckwok2/research/segmentation2/Skipped_Images'         # Final Fails
MODEL_PATH    = '/home/hlckwok2/research/segmentation2/pupil_unet_model.h5'
TARGET_SIZE   = (256, 256)

# Setup Folders - Wipes old results for a clean start


# --- 3. DATA & MODEL UTILITIES ---
def load_data(img_dir, mask_dir):
    images, masks = [], []
    
    # 1. Get all files in both directories
    img_list = [f for f in os.listdir(img_dir) if not f.startswith('.')]
    mask_list = [f for f in os.listdir(mask_dir) if not f.startswith('.')]
    
    log(f"Scanning: {len(img_list)} images found, {len(mask_list)} masks found.")

    # 2. Create a lookup dictionary for masks (Base Name -> Full Filename)
    # This handles 'mask_eye1.png', 'eye1.png', 'EYE1.JPG' etc.
    mask_map = {}
    for m in mask_list:
        base = os.path.splitext(m)[0].lower()
        # Remove common prefixes like 'mask_' to unify the names
        clean_base = base.replace('mask_', '')
        mask_map[clean_base] = m

    # 3. Match images to masks
    for f in img_list:
        img_base = os.path.splitext(f)[0].lower()
        clean_img_base = img_base.replace('mask_', '') # Just in case

        if clean_img_base in mask_map:
            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, mask_map[clean_img_base])
            
            img_cv = cv2.imread(img_path)
            mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img_cv is not None and mask_cv is not None:
                img_cv = cv2.resize(img_cv, TARGET_SIZE) / 255.0
                mask_cv = cv2.resize(mask_cv, TARGET_SIZE) / 255.0
                images.append(img_cv)
                masks.append(np.expand_dims(mask_cv, axis=-1))
            else:
                print(f"[ERROR]: Failed to read {f} or its mask.")
        else:
            print(f"[MISSING]: No mask found for {f}")

    log(f"Successfully paired {len(images)} out of {len(img_list)} images.")
    return np.array(images), np.array(masks)

def build_unet():
    inputs = layers.Input((256, 256, 3))
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    u2 = layers.UpSampling2D()(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u2)
    output = layers.Conv2D(1, 1, activation='sigmoid')(c4)
    return models.Model(inputs, output)

# --- 4. THE TRIPLE-PASS SEGMENTATION ENGINE ---
def run_segmentation(img_path, model, mode="STRICT"):
    try:
        orig = cv2.imread(img_path)
        if orig is None: return False
        
        h, w = orig.shape[:2]
        proc = cv2.resize(orig, TARGET_SIZE) / 255.0
        pred = model.predict_on_batch(np.expand_dims(proc, axis=0))[0]
        
        # PARAMETERS PER PHASE
        if mode == "STRICT":
            thresh, sol_min, margin_p, k_size = 0.50, 0.45, 0.05, None
        elif mode == "RESCUE":
            thresh, sol_min, margin_p, k_size = 0.15, 0.25, 0.02, 7
        else: # ULTRA
            thresh, sol_min, margin_p, k_size = 0.08, 0.10, 0.01, 15

        raw_mask = (cv2.resize(pred, (w, h)) > thresh).astype(np.uint8) * 255
        
        if k_size:
            kernel = np.ones((k_size, k_size), np.uint8)
            raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            
            # Geometry Math
            circle_area = np.pi * (radius ** 2)
            solidity = area / circle_area if circle_area > 0 else 0
            margin = w * margin_p
            on_edge = (cx < margin or cx > w-margin or cy < margin or cy > h-margin)

            if radius > (w * 0.04) and not on_edge and solidity > sol_min:
                # SUCCESS
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)
                
                white_bg = np.full(orig.shape, 255, dtype=np.uint8)
                res = cv2.add(cv2.bitwise_and(orig, orig, mask=mask), 
                             cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask)))
                
                # Dynamic Save Path
                folders = {"STRICT": OUTPUT_DIR, "RESCUE": RESCUED_DIR, "ULTRA": ULTRA_DIR}
                rel = os.path.relpath(os.path.dirname(img_path), INFERENCE_DIR if mode=="STRICT" else SKIPPED_DIR)
                target = os.path.join(folders[mode], rel)
                os.makedirs(target, exist_ok=True)
                cv2.imwrite(os.path.join(target, os.path.basename(img_path)), res)
                
                # Cleanup skipped folder if rescued
                if mode != "STRICT" and os.path.exists(img_path):
                    os.remove(img_path)
                return True
            elif mode == "ULTRA":
                print(f"[ULTRA-REJECT]: {os.path.basename(img_path)} | Sol:{solidity:.2f} | Rad:{radius:.1f}")

        # If strict fails, copy image for next pass
        if mode == "STRICT":
            rel = os.path.relpath(os.path.dirname(img_path), INFERENCE_DIR)
            target = os.path.join(SKIPPED_DIR, rel)
            os.makedirs(target, exist_ok=True)
            shutil.copy(img_path, os.path.join(target, os.path.basename(img_path)))
        return False
    except Exception as e:
        print(f"Error on {img_path}: {e}")
        return False

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    log("Starting Optimized JPG Pupil Pipeline")

    # LOAD OR TRAIN
    if os.path.exists(MODEL_PATH):
        log(f"Loading existing model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        log("No saved model found. Training fresh U-Net...")
        X, y = load_data(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
        model = build_unet()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=50, batch_size=8, verbose=1)
        model.save(MODEL_PATH)
        log(f"Model saved to {MODEL_PATH}")

    # PASS 1: STRICT
    all_imgs = [os.path.join(r, f) for r, _, fs in os.walk(INFERENCE_DIR) for f in fs if f.lower().endswith(('.jpg', '.jpeg'))]
    log(f"Phase 1: Strict Processing ({len(all_imgs)} images)...")
    with ThreadPoolExecutor(max_workers=NUM_CORES) as ex:
        list(tqdm(ex.map(lambda p: run_segmentation(p, model, "STRICT"), all_imgs), total=len(all_imgs), file=sys.stdout))

    # PASS 2 & 3: RESCUE & ULTRA
    for mode in ["RESCUE", "ULTRA"]:
        skipped = [os.path.join(r, f) for r, _, fs in os.walk(SKIPPED_DIR) for f in fs if f.lower().endswith(('.jpg', '.jpeg'))]
        if not skipped: 
            log(f"No images left for {mode} phase.")
            continue
        log(f"Phase: {mode} on {len(skipped)} images...")
        with ThreadPoolExecutor(max_workers=NUM_CORES) as ex:
            list(tqdm(ex.map(lambda p: run_segmentation(p, model, mode), skipped), total=len(skipped), file=sys.stdout))

    # CONSOLIDATION
    log("Merging Rescued and Ultra results into 'Segmented_Results'...")
    for rescue_folder in [RESCUED_DIR, ULTRA_DIR]:
        for root, _, files in os.walk(rescue_folder):
            for f in files:
                rel = os.path.relpath(root, rescue_folder)
                dest = os.path.join(OUTPUT_DIR, rel)
                os.makedirs(dest, exist_ok=True)
                shutil.copy(os.path.join(root, f), os.path.join(dest, f))

    log("Workflow Complete. Final Fails are in 'Skipped_Images'.")