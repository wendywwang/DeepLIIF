import os
root_folder=os.getenv("DATA_DIR")
output_model_folder = os.environ["RESULT_DIR"]
output_model_path = os.path.join(output_model_folder,"model")
output_model_path_file = os.path.join(output_model_path,"trained_model.pt")
output_model_path_onnx = os.path.join(output_model_path,"trained_model.onnx")


import subprocess
print('Install packages and start gpu monitor ...')
subprocess.Popen('pip install dominate visdom gpustat numba --user; pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html; bash monitor_gpu.sh',shell=True)

print('Sleep for 10s to ensure package installation is completed')
subprocess.run('sleep 10',shell=True)

import psutil

if __name__ == '__main__':
    print('-------- start training... --------')
    subprocess.run(f'python cli.py train --dataroot {root_folder} --name Test_Model --remote True --pickle_transfer_cmd custom_save.save_to_storage_volume --batch_size 2 --display_env $APP_ID',shell=True)

    # (DP) use batch_size > 1 if you want to leverage multiple gpus; batch_size=1 will only effectively use 1 gpu, because this setting is picked up by DP's single process and the amount is distributed across multiple gpus
    # (DDP) even if batch size is 1, each process/gpu will get one image per mini batch. because this setting is picked up by each of DDP's processes, one process having 1 gpu
    
    for process in psutil.process_iter():
        if process.cmdline() == ['bash', 'monitor_gpu.sh']:
            print('Terminating gpu monitor...')
            process.terminate()
            break