#!/bin/bash

# 所有标签和数据集定义
all_labels=('1' '2_1' '2_2' '3_1' '3_2' '3_3' '3_4' '4' '5_1' '5_2' '5_3' '5_4')
all_datasets=('sift' 'gist' 'glove100' 'msong' 'enron' 'audio')

# 默认操作类型
operation="all"

# 处理参数
if [ $# -eq 0 ]; then
  datasets=("${all_datasets[@]}")
  labels=("${all_labels[@]}")
elif [ $# -eq 1 ]; then
  if [[ "$1" == "run" || "$1" == "plot" || "$1" == "all" ]]; then
    operation="$1"
    datasets=("${all_datasets[@]}")
    labels=("${all_labels[@]}")
  else
    dataset="$1"
    if [[ ! " ${all_datasets[*]} " =~ " $dataset " ]]; then
      echo "❌ 无效的数据集：$dataset，支持的有：${all_datasets[*]}"
      exit 1
    fi
    datasets=("$dataset")
    labels=("${all_labels[@]}")
  fi
elif [ $# -eq 2 ]; then
  if [[ "$1" == "run" || "$1" == "plot" || "$1" == "all" ]]; then
    operation="$1"
    dataset="$2"
    if [[ ! " ${all_datasets[*]} " =~ " $dataset " ]]; then
      echo "❌ 无效的数据集：$dataset，支持的有：${all_datasets[*]}"
      exit 1
    fi
    datasets=("$dataset")
    labels=("${all_labels[@]}")
  else
    dataset="$1"
    label="$2"

    if [[ ! " ${all_datasets[*]} " =~ " $dataset " ]]; then
      echo "❌ 无效的数据集：$dataset，支持的有：${all_datasets[*]}"
      exit 1
    fi

    if [[ ! " ${all_labels[*]} " =~ " $label " ]]; then
      echo "❌ 无效的标签：$label，支持的有：${all_labels[*]}"
      exit 1
    fi

    datasets=("$dataset")
    labels=("$label")
  fi
elif [ $# -eq 3 ]; then
  operation="$1"
  dataset="$2"
  label="$3"

  if [[ "$operation" != "run" && "$operation" != "plot" && "$operation" != "all" ]]; then
    echo "❌ 无效的操作：$operation，应为 run / plot / all"
    exit 1
  fi

  if [[ ! " ${all_datasets[*]} " =~ " $dataset " ]]; then
    echo "❌ 无效的数据集：$dataset，支持的有：${all_datasets[*]}"
    exit 1
  fi

  if [[ ! " ${all_labels[*]} " =~ " $label " ]]; then
    echo "❌ 无效的标签：$label，支持的有：${all_labels[*]}"
    exit 1
  fi

  datasets=("$dataset")
  labels=("$label")
else
  echo "❌ 用法错误：支持格式如下："
  echo "    ./run_all.sh"
  echo "    ./run_all.sh <dataset>"
  echo "    ./run_all.sh <dataset> <label>"
  echo "    ./run_all.sh <operation: run|plot|all> [<dataset>] [<label>]"
  exit 1
fi

# 执行
for dataset in "${datasets[@]}"; do
  for label in "${labels[@]}"; do
    full_dataset="${dataset}label_${label}"
    echo "===== Running experiment on ${full_dataset} ====="

    if [[ "$operation" == "run" || "$operation" == "all" ]]; then
      echo ">>> [RUN] python run.py"
      python ../../algorithm/Puck/run.py --algorithm puck --neurips23track filter --dataset "${full_dataset}"
    fi

    if [[ "$operation" == "plot" || "$operation" == "all" ]]; then
      echo ">>> [PLOT] python plot.py"
      python ../../algorithm/Puck/plot.py --dataset "${dataset}${label}" --neurips23track filter
    fi

    echo ""
  done
done
