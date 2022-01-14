import os
root_folder='/mnts/deepliif-data/DeepLIIF_Datasets'#os.getenv("DATA_DIR")
# output_model_folder = os.environ["RESULT_DIR"]
# output_model_path = os.path.join(output_model_folder,"model")
# output_model_path_file = os.path.join(output_model_path,"trained_model.pt")
# output_model_path_onnx = os.path.join(output_model_path,"trained_model.onnx")


import subprocess
print('Install packages...')
subprocess.run('pip install dominate visdom gpustat --user',shell=True)

# print('Detecting keyword DLI in env vars from python...')
# subprocess.run('env | grep DLI | wc -l',shell=True)

# print('Starting gpu monitor...')
# subprocess.Popen('bash monitor_gpu.sh',shell=True)


if __name__ == '__main__':
    print('-------- os.environ ---------')
    print(os.environ)
    print('-------- ls -lh --------')
    subprocess.run('ls -lh',shell=True)
    print('-------- start training... --------')
    subprocess.run(f'python train.py --dataroot {root_folder} --name Test_Model --model DeepLIIF --remote True --pickle_transfer_cmd custom_save.save_to_storage_volume --gpu_ids 0',shell=True)
