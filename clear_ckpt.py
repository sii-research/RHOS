import os
import glob

def clear_ckpt_files(task_name):
    """
    遍历 base_dir 下所有符合 data/outputs/{task_name}/flow_{x}/run_{y}/checkpoints/ 的路径，
    将其中所有 .ckpt 文件清空（保留文件名，内容变为空）。
    """
    base_dir="data/outputs"
    # 构造通配符路径：支持任意 task_name、flow_x、run_y
    pattern = os.path.join(base_dir, task_name, "flow_*", "run_*", "checkpoints", "*.ckpt")
    ckpt_files = glob.glob(pattern, recursive=True)

    if not ckpt_files:
        print("未找到任何 .ckpt 文件。")
        return

    print(f"找到 {len(ckpt_files)} 个 .ckpt 文件，正在清空内容...")

    for ckpt_path in ckpt_files:
        try:
            # 打开文件并清空内容（保留文件）
            with open(ckpt_path, 'w') as f:
                pass  # 写入空内容
            print(f"已清空: {ckpt_path}")
        except Exception as e:
            print(f"处理文件时出错: {ckpt_path} - {e}")

if __name__ == "__main__":
    task_name="pusht"
    clear_ckpt_files(task_name)