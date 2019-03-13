# How to use cluster framework to conduct hyper-parameter selction

## Do not panic: a brief introduction
- two tutorials to read:
  - [Original Experiment Suite PDF](https://github.com/rueckstiess/expsuite/blob/master/documentation.pdf)
  - [Gregor's Git](https://github.com/gregorgebhardt/cluster_work)

## Quick start
- Download Gregor's git and create a new python script which import cluster_work
- Install yaml (`pip install PyYAML`) and mpi4py and cloudpickle. **Attention**: Create a pure py3.6 virtual  environment unless mpi4py in python 2.7 global will disturb the usage in the python 3.6.
- Create your own 'yaml' file.
- Define your python sentences under 'iteration'. The return of the funtion 'iteration' will be saved under the default path.
- **Parameters passing**: config will be a dictionary which gets values from yaml file. So just write like `config['params']['b_value']`
- Add init function:
  ```python
     def __init__(self):
        ClusterWork.__init__(self)
        self.train_obj = DQNTrain()
  ```

- reset只在repitition的时候重置，每一次iteration的时候不调用，repetition真的只是把你的代码重复跑一次
- 然后对于list中的参数，会针对没一个参数，跑若干次repetitions
