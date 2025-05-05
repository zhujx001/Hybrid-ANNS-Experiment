import os
import sys
import subprocess
import stat

def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 得到 script/Faiss 路径
    project_root = os.path.dirname(os.path.dirname(script_dir))  # 回退到项目根目录
    build_script = os.path.join(project_root, "algorithm", "Faiss", "build.sh")
    
    # 确保构建脚本有执行权限
    if os.path.exists(build_script):
        current_mode = os.stat(build_script).st_mode
        os.chmod(build_script, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    else:
        print(f"错误: 构建脚本不存在: {build_script}")
        return 1
    
    # 使用bash明确执行脚本，而不是直接执行
    print("Building project...")
    subprocess.run(["bash", build_script], check=True)
    
    # List of executables to run - modify this list as needed
    executables_to_run = [
        {
            "name": "test_getIndex", 
            "type": "label",
            "args": ["-s", "1"],
            "use_taskset": True,
            "cpu_cores": "0-31"  # 使用0-31核心
        },
        # 单标签 非批量 16线程
        {
            "name": "test_oneattr_nobatch_or_batch", 
            "type": "label",
            "args": ["-s", "1", "-b", "0", "-t", "16", "-c", "2"], 
            "use_taskset": False  # 不使用taskset
        },
        # 单标签 非批量 1线程
        {
            "name": "test_oneattr_nobatch_or_batch", 
            "type": "label",
            "args": ["-s", "1", "-b", "0", "-t", "1", "-c", "2"], 
            "use_taskset": False  # 不使用taskset
        },
        # 单标签 批量 1线程
        {
            "name": "test_oneattr_nobatch_or_batch", 
            "type": "label",
            "args": ["-s", "1", "-b", "1", "-t", "16", "-c", "2"], 
            "use_taskset": True,
            "cpu_cores": "0"
        },
        # 三标签 非批量 16线程
        {
            "name": "test_threeattr_nobatch_or_batch", 
            "type": "label",
            "args": ["-s", "1", "-b", "0", "-t", "16", "-c", "2"], 
            "use_taskset": False  # 不使用taskset
        },
        # 三标签 非批量 1线程
        {
            "name": "test_threeattr_nobatch_or_batch", 
            "type": "label",
            "args": ["-s", "1", "-b", "1", "-t", "16", "-c", "2"], 
            "use_taskset": True,
            "cpu_cores": "0-15"
        },
        # 三标签 批量 1线程
        {
            "name": "test_threeattr_nobatch_or_batch", 
            "type": "label",
            "args": ["-s", "1", "-b", "1", "-t", "1", "-c", "2"], 
            "use_taskset": True,
            "cpu_cores": "0"
        },
        # 范围查询 索引
        {
            "name": "test_getIndex", 
            "type": "range",
            "args": ["-s", "1"],
            "use_taskset": True,
            "cpu_cores": "0"  # 使用0核心
        },
        # 范围查询 非批量 1线程
        {
            "name": "test_range_nobatch_or_batch", 
            "type": "range",
            "args": ["-s", "1", "-b", "0", "-t", "1", "-c", "2"], 
            "use_taskset": False  # 不使用taskset
        },
    ]
    
    # Run each executable
    for exe in executables_to_run:
        exe_path = os.path.join(project_root, "algorithm", "Faiss", "build", "bin", exe["type"], exe["name"])
        if os.path.exists(exe_path):
            # 准备命令行
            if exe.get("use_taskset", False):
                cmd = ["taskset", "-c", exe.get("cpu_cores", "0-15"), exe_path]
                cmd_display = f"taskset -c {exe.get('cpu_cores', '0-15')} {exe_path}"
            else:
                cmd = [exe_path]
                cmd_display = exe_path
                
            # 添加命令行参数
            if exe.get("args"):
                cmd.extend(exe.get("args"))
                cmd_display += " " + " ".join(exe.get("args"))
            
            # 执行命令
            print(f"\nRunning: {cmd_display}")
            print("-" * 50)
            subprocess.run(cmd)
            print("-" * 50)
        else:
            print(f"Error: Executable {exe['name']} not found in {exe_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())