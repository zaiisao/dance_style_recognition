import os
import glob
import argparse
import torch
import torch.multiprocessing as mp
import time
from queue import Empty

from process_lma_features import process_single_video

# Import the MoGe model class
from moge.model.v2 import MoGeModel

def worker_process(gpu_id, queue, output_dir, viz):
    """
    1. Initializes models on the assigned GPU (Run once per worker)
    2. Consumes files from the queue until empty
    """
    process_name = mp.current_process().name

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = f"cuda:0"
    print(f"[{process_name}] Launching on GPU {str(gpu_id)}...")

    # --- A. LOAD MODELS ONCE ---
    try:
        # Load NLF
        nlf_model = torch.jit.load('models/nlf_l_multi_0.3.2.torchscript', map_location=device).eval()
        # Load MoGe
        with torch.cuda.device(device):
            moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
        print(f"[{process_name}] Models loaded. Ready to work.")
    except Exception as e:
        print(f"[{process_name}] CRITICAL MODEL LOAD FAIL: {e}")
        return

    # --- B. PROCESSING LOOP ---
    processed_count = 0
    while True:
        try:
            # Get next file from queue (timeout allows clean exit if queue stuck)
            video_path = queue.get(timeout=5)
        except Empty:
            # Queue is empty, work is done
            break

        try:
            # Run single-video processing
            process_single_video(
                video_path=video_path, 
                output_dir=output_dir, 
                nlf_model=nlf_model, 
                moge_model=moge_model, 
                device=device, 
                viz=viz
            )
            processed_count += 1
        except Exception as e:
            print(f"[{process_name}] FAILED on {os.path.basename(video_path)}: {e}")
        finally:
            # Signal that this specific task is complete
            queue.task_done()

    print(f"[{process_name}] Done. Processed {processed_count} videos.")

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Master Runner")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing .mp4 files")
    parser.add_argument("--input_file", type=str, help="Optional: .txt file containing specific filenames to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save results")
    parser.add_argument("--workers_per_gpu", type=int, default=6, 
                        help="Number of parallel processes per GPU. (Default: 6)")
    parser.add_argument("--viz", action="store_true", help="Enable debug video generation")
    args = parser.parse_args()

    # 1. Setup
    mp.set_start_method('spawn', force=True) # Critical for CUDA
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Collect Videos & Filter Existing
    all_files = []
    if args.input_file:
        print(f"Reading video list from {args.input_file}...")
        with open(args.input_file, 'r') as f:
            # Read lines, strip whitespace, and ignore empty lines
            filenames = [line.strip() for line in f if line.strip()]
            for name in filenames:
                # If the file in the TXT isn't an absolute path, join it with input_dir
                full_path = name if os.path.isabs(name) else os.path.join(args.input_dir, name)
                all_files.append(full_path)
    else:
        print(f"Scanning {args.input_dir}...")
        all_files = glob.glob(os.path.join(args.input_dir, "*.mp4"))
    
    # Filter list: Only keep videos where the output .npy doesn't exist yet
    video_files = []
    for v in all_files:
        if not os.path.exists(v):
            print(f"Warning: File not found, skipping: {v}")
            continue
        base_name = os.path.splitext(os.path.basename(v))[0]
        expected_npy = os.path.join(args.output_dir, f"{base_name}_features.npy")
        
        if not os.path.exists(expected_npy):
            video_files.append(v)

    total_files = len(video_files)
    skipped_count = len(all_files) - total_files

    print(f"Found {len(all_files)} total videos.")
    print(f"Skipping {skipped_count} already processed.")
    print(f"Queueing {total_files} new videos.")

    if total_files == 0:
        print("All files up to date. Exiting.")
        return

    # 3. Fill Queue
    task_queue = mp.JoinableQueue()
    for v in video_files:
        task_queue.put(v)

    # 4. Launch Workers
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")
    if num_gpus == 0:
        print("No GPUs found! Exiting.")
        return

    workers = []
    print(f"Spawning {args.workers_per_gpu} workers per GPU ({num_gpus * args.workers_per_gpu} total)...")
    
    for gpu_id in range(num_gpus):
        for _ in range(args.workers_per_gpu):
            p = mp.Process(
                target=worker_process, 
                args=(gpu_id, task_queue, args.output_dir, args.viz)
            )
            p.start()
            workers.append(p)

    # 5. Monitor
    print("\n--- PROCESSING STARTED ---")
    start_time = time.time()
    
    try:
        # Wait for queue to be empty
        while not task_queue.empty():
            remaining = task_queue.qsize()
            done = total_files - remaining
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            
            print(f"Progress: {done}/{total_files} | Rate: {rate:.2f} vids/sec", end='\r')
            time.sleep(5)
            
        # Final join to ensure all items are actually processed
        task_queue.join()
        
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected. Terminating workers...")
        for p in workers:
            p.terminate()
            
    print(f"\n\nDone! Processed {total_files} videos in {(time.time() - start_time)/60:.1f} minutes.")

if __name__ == "__main__":
    main()
