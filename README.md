# Requirements

    - python version >= 3.8.0
    - git 2.44.0

! note if you are using windows much better to use the gui bash of git 

### Building python environment
    - pip install virtualenv
    - Check that virtualenv is installed through the command pip --version
    - Install virtualenvwrapper-win through the command pip install virtualenvwrapper-win

## <div>How to use</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ppogg/YOLOv5-Lite/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):


```bash
$ python3 -m venv <name>
$ source <name>/bin/activate

$ git clone https://github.com/supremoods/obs-detection.git
$ cd obs-detection
$ pip install -r requirements.txt
```

</details>


<details open>
<summary>Training</summary>

```bash
$ python train.py --data coco.yaml --cfg v5lite-e.yaml --weights v5lite-e.pt --batch-size 64                                     
```

 If you use multi-gpu. It's faster several times:
  
 ```bash
$ python -m torch.distributed.launch --nproc_per_node 2 train.py
```
</details>  


## References
    For the calculation of measurement distance detection watch the video

    [**Youtube**](https://www.youtube.com/watch?v=zzJfAw3ASzY&list=PLJ958Ls6nowWdRrQqmmUbDUtlUgGR0Qh_&index=2) for arm-cpu