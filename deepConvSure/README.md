# deepConvSurv_PyTorch
Implementation of deepConvSurv in PyTorch

To run the code
```
python -u cluster_select_deepconvsurv.py | tee -a /path/to/save/log.txt
```

required modification 
```
img_path='/path/to/all/pathces'
label_path = '/path/to/label/file' # contains surv and status of each WSIs
expand_label_path = '/path/to/extend/label/file/for/each/cluster' 
```
