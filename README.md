# HybridANNS

---

## 📦 Dataset Download

Datasets used in the experiments can be downloaded from the following link:  
👉 *[https://drive.google.com/file/d/1eYBPsOP9JJd7zVjpXUFnWhNIGfw-FZid/view?usp=drive_link]*

> ⚠️ After downloading, **make sure to place the extracted files in the following directories**:
> 
> ```
> HybridANNS/data/Experiment/
> HybridANNS/data/temp/
> ```

---

## 📚 Algorithm Implementations

Each hybrid filtering algorithm is implemented under the `algorithm/` directory.  
To reproduce experiments for a specific algorithm, please refer to its respective `README.md` file:

```
algorithm/
├── SeRF/
│   └── README.md
├── iRange/
│   └── README.md
├── WinFilter/
│   └── README.md
...
```

---

## 🚀 Running the Code

If you want to quickly run the experiments, you can use the scripts provided under the `script/` directory for one-click execution:

```
script/
└── DSG/
    └── run.py
```

> ⚠️ **Important Notes Before Running:**
> 
> - Please carefully read the code beforehand to understand its logic.
> - Many algorithms require proper environment setup.  
>   It is strongly recommended to first read the `README.md` files under each algorithm folder and complete the required environment configuration.
> - Make sure the dataset paths are correctly set as mentioned above.
> 
> If you encounter any problems, you can refer to the original code repository for more details.
