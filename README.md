# DPVP
Source code of KDD23 Paper "Modeling Dual Period-Varying Preferences for Takeaway Recommendation"

### Data

**source**: download MT-small data from https://drive.google.com/drive/folders/1k9q9X4lK-pnJq2PaK6YtqSOCPOSU-O4D?usp=drive_link

**structure**:

​	|code

​		|mygraph

​		conf.py

​		main.py

​		utils.py

​	|MT-small

**data format**:

- Train_data

| user | res  | itemlst                          | label | hour |
| ---- | ---- | -------------------------------- | ----- | ---- |
| 0    | 0    | [1]                              | 1     | 14   |
| 1    | 1    | [2, 249, 291, 143, 53, 317, 126] | 1     | 22   |
| 2    | 2    | [3, 195, 204, 468]               | 1     | 12   |

- Valid data or test data. For each ground truth valid/test data, we randomly select 99 stores that the user has not interacted with, serving as negative samples. Records sharing the same 'sample num' indicate that they are either a single positive sample or the negative samples derived based on the positive sample.

| user | Res  | itemlst              | label | hour | sample_num |
| ---- | ---- | -------------------- | ----- | ---- | ---------- |
| 7    | 7    | [8, 244]             | 1     | 18   | 0          |
| 7    | 4054 | [2030, 1743, 385]    | 0     | 18   | 0          |
| 7    | 2183 | [687, 381, 515, 669] | 0     | 18   | 0          |



```
cd code
python3 main.py
```

### reference 

```
@inproceedings{zhang2023modeling,
author = {Zhang, Yuting and Wu, Yiqing and Le, Ran and Zhu, Yongchun and Zhuang, Fuzhen and Han, Ruidong and Li, Xiang and Lin, Wei and An, Zhulin and Xu, Yongjun},
title = {Modeling Dual Period-Varying Preferences for Takeaway Recommendation},
booktitle={the 29th SIGKDD conference on Knowledge Discovery and Data Mining (KDD 2023)},
pages = {5628–5638},
numpages = {11},
year={2023}
}
```

