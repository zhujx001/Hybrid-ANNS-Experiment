## 环境要求

- CMake 3.10+
- C++17兼容的编译器
- BLAS/LAPACK库
- OpenMP

## 构建

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)

# 或直接# 执行构建脚本
./build.sh
```

## 运行

```bash
python3 run_faiss.py
```
