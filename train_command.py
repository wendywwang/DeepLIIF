import os
root_folder=os.getenv("DATA_DIR")
output_model_folder = os.environ["RESULT_DIR"]
output_model_path = os.path.join(output_model_folder,"model")
output_model_path_file = os.path.join(output_model_path,"trained_model.pt")
output_model_path_onnx = os.path.join(output_model_path,"trained_model.onnx")


import subprocess
print('Install packages and start gpu monitor ...')
subprocess.Popen('pip install dominate visdom gpustat --user; bash monitor_gpu.sh',shell=True)

print('Sleep for 10s to ensure package installation is completed')
subprocess.run('sleep 10',shell=True)

import psutil

if __name__ == '__main__':
    print('-------- start training... --------')
    subprocess.run(f'python cli.py train --dataroot {root_folder} --name Test_Model --model DeepLIIF --remote True --pickle_transfer_cmd custom_save.save_to_storage_volume --gpu_ids 0 --batch_size 2',shell=True)

    # use batch_size > 1 if you want to leverage multiple gpus; batch_size=1 will only effectively use 1 gpu
    
    for process in psutil.process_iter():
        if process.cmdline() == ['bash', 'monitor_gpu.sh']:
            print('Terminating gpu monitor...')
            process.terminate()
            break