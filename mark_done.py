import os
import sys

def mark_range_done(start: int, end: int, base_name: str = "TASKS.jsonl"):
    lock_dir = "locks"
    for idx in range(start, end):
        locked_path = os.path.join(lock_dir, f"{base_name}.{idx}.locked")
        done_path = os.path.join(lock_dir, f"{base_name}.{idx}.done")

        if os.path.exists(locked_path):
            os.rename(locked_path, done_path)
            print(f"✅ Renamed: {locked_path} → {done_path}")
        else:
            print(f"⏭️  Skipped (no .locked file): {locked_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mark_range_done.py <start_index> <end_index>")
        print("Example: python mark_range_done.py 10 20")
        sys.exit(1)

    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
        if start >= end:
            raise ValueError("start must be less than end")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    mark_range_done(start, end)