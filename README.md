# YouTube Sentiment Analysis

This repository contains code for performing sentiment analysis on YouTube comments using a BERT-based model.

👉 **GitHub Repository**: [https://github.com/CHEN010325/YouTube-sentiment-analysis](https://github.com/CHEN010325/YouTube-sentiment-analysis)
If you have any questions, email sl695969@outlook.com
---

## 📦 Dependencies

To reproduce the results **exactly**, please use the following package versions:

- pandas==2.2.3  
- numpy==2.1.2  
- matplotlib==3.10.1  
- transformers==4.51.3  
- tqdm==4.67.1  
- torch==2.7.0+cu128  
- scikit-learn==1.6.1  


---

## 🔁 Reproducibility

> ✅ To **fully reproduce** the results:
> 
> - Use the **exact versions** listed above  
> - Use **Python seed = 42** (already set in the code)  
> - 🖥️ **Tested on**: NVIDIA GeForce RTX 5070 Ti with CUDA 11.8 (Runtime: 3m34s)




### 🔒 Fixed random seed:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False**
