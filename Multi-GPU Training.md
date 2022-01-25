# Multi-GPU Training

There are 2 ways you can leverage multiple GPUs to train DeepLIIF, **Data Parallel (DP)** or **Distributed Data Parallel (DDP)**. Both cases are a kind of **data parallelism** supported by PyTorch.

The key difference is that DP is **single process multi-threading** while DDP can have **multiple processes**.

**TL;DR**

Use DP if you
- are used to the way to train DeepLIIF on multiple GPUs since its first release, OR
- do **not** have multiple GPU machines to utilize, OR
- are fine with the training being a bit slower

Use DDP if you
- are willing to try a slightly different way to launch the training than before, OR
- do have multiple GPU machines for cross-node distributino, OR
- want to get as fast training as possible


## Data Parallel (DP)
DP is single-process. It means that **all the GPUs you want to use must be on the same machine** so that they can be included in the same process - you cannot distribute the training across multiple GPU machines, unless you write your own code to handle inter-node (node = machine) communication.

To split and manage the workload for multiple GPUs within the same process, DP uses multi-threading. 

It is worth noting that multi-threading in this case can lead to significant performance overhead, and slow down your training. See a short discussion in [PyTorch's CUDA Best Practices](https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel).

### Train with DP
Example with 2 GPUs (of course on 1 machine):
```
python cli.py train --dataroot <data_dir> --batch-size 6 --gpu-ids 0 --gpu-ids 1
```
Note that
1. `batch-size` is defined per process. Since DP is a single-process method, the `batch-size` you set is the **effective** batch size.

## Distributed Data Parallel (DDP)
DDP usually spawns multiple processes. 

**DeepLIIF's code follows the PyTorch recommendation to spawn 1 process per GPU** ([doc](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md#application-process-topologies)). If you want to assign multiple GPUs to each process, you will need to make modifications to DeepLIIF's code (see [doc](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#combine-ddp-with-model-parallelism)).


### Train with DDP
#### 1. Local Machine
To launch training using DDP on a local machine, simply add `torchrun --nproc_per_node <gpu_num>` before the training command. Example with 2 GPUs (on 1 machine):
```
torchrun --nproc_per_node 2 cli.py train --dataroot <data_dir> --batch-size 3 --gpu-ids 0 --gpu-ids 1
```
Note that
1. `batch-size` is defined per process. Since DDP is a single-process method, the `batch-size` you set is the batch size for each process, and the **effective** batch size will be `batch-size` multiplied by the number of processes you started. In the above example, it will be 3 * 2 = 6.
2. You still need to provide **all GPU ids to use** to the training command. Internally, in each process DeepLIIF picks the device using `gpu_ids[local_rank]`. If you provide `--gpu-ids 2 --gpu-ids 3`, the process with local rank 0 will use gpu id 2 and that with local rank 1 will use gpu id 3. 
3. `-t 3 --log_dir <log_dir>` is not required, but is a useful setting in `torchrun` that saves the log from each process to your target log directory. For example:
```
torchrun -t 3 --log_dir <log_dir> --nproc_per_node 2 cli.py train --dataroot <data_dir> --batch-size 3 --gpu-ids 0 --gpu-ids 1
```

#### 2. Kubernetes-Based Training Service
To launch training using DDP on a kubernetes-based service where each process will have its own pod and a dedicated GPU, and there is an existing task manager/scheduler in place, you may submit a script with training command like the following:
```
import os
import torch.distributed as dist
def init_process():
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://' + os.environ['MASTER_ADDR'] + ':' + os.environ['MASTER_PORT'],
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE']))

root_folder = <data_dir>

if __name__ == '__main__':
    init_process()
    subprocess.run(f'python cli.py train --dataroot {root_folder} --remote True --batch-size 3 --gpu-ids 0',shell=True)
```
Note that
1. Always provide `--gpu-ids 0` to the training command for each process/pod if the gpu id gets re-named in each pod. If not, you will need to pass the correct gpu id in a dynamic way, possibly through an environment variable in each pod.

#### 3. Multiple Virtual Machines
To launch training across multiple VMs, you can refer to the scheduler framework you use. For each process, similar to the example for kubernetes, you will need to initiate the process group so that the current process knows who it is, where are its peers, etc., and then execute the regular training command in a subprocess.

## Move from Single-GPU to Multi-GPU: Impact on Hypter-Parameters
To achieve equivalently good training results, you may want to adjust some hypter-parameters you figured out for a single GPU training.

### Batch Size & Learning Rate
Backward propagation by default runs at the end of every batch to find how much changes to make in parameters. An immediate outcome from using multiple GPUs is that we have a larger effective batch size. 

In DDP, this means fewer gradient descent because DDP averages the gradients from all processes ([doc](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)). Assume thats in 1 epoch, a single-GPU training does gradient descent for 200 times. Now with 2 GPUs/processes, the training will have 100 batches in each process and does the gradient descent using the averaged gradients of the 2 GPUs/processes, one for each batch, which is 100 times.

You may want to compensate this by increasing the learning rate proportionally.

DP is slightly different, in that it sums up the gradients from all GPUs/threads ([doc](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel)). However, practically the performance (accuracy) still suffers from the larger effective batch size, which can be mitigated by increasing the learning rate.