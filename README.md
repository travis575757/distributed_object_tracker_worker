
[Distributed object tracker worker application](https://github.com/travis575757/distributed_object_tracker)

---

# Installation

#### Create environment and activate
```bash
conda create --name pysot python=3.7
conda activate pysot
```

#### Install numpy/pytorch/opencv
```
conda install numpy
conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
pip install opencv-python
```

#### Install other requirements
```
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX zmq gdown
```

#### Build extensions
```
python setup.py build_ext --inplace
```

# Usage

```
python spawn.py --server_ip <server ip> --gpu_id <gpu id>
```

Where server ip is the coordinating server ip address and gpu id is the gpu index value to use for this worker

#### Example
```
python spawn.py --server_ip 0.0.0.0 --gpu_id 0
```
