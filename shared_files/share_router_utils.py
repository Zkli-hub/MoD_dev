"""
INSTALL:

pip install cos-python-sdk-v5 rich

"""
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import time
import shutil
import torch

class UploadFile(object):
    def __init__(self, progress, bucket_name):
        self.progress_value = 0
        self.progress = progress
        self.upload_tqdm = self.progress.add_task(description="Progress", total = 100)
        self.bucket_name = bucket_name

    def __call__(self, client, src_file_path, console):
        def callback_interface(complete, total):
            step = int(round(complete * 100 / total)) - self.progress_value
            self.progress.advance(self.upload_tqdm, advance = step)
            self.progress_value = int(round(complete * 100 / total))

        if os.path.exists(src_file_path) == False:
            print('Not exited file')
            return
        else:
            base_fname = os.path.basename(src_file_path)
            for i in range(0, 10):
                try:
                    response = client.upload_file(
                        Bucket=self.bucket_name,
                        Key = base_fname,
                        LocalFilePath = src_file_path,
                        EnableMD5 = False,
                        progress_callback = callback_interface
                    )
                    break
                except:
                    print('Error')
            return

class DownloadFile(object):
    def __init__(self, progress, bucket_name, fold_path):
        self.progress_value = 0
        self.progress = progress
        self.download_tqdm = self.progress.add_task(description="Progress", total = 100)
        self.fold_path = fold_path
        self.bucket_name = bucket_name
    
    def check_file_exist(self, client, remote_file_name):
        response = client.object_exists(
                    Bucket=self.bucket_name,
                    Key = remote_file_name)
        return response
        

    def __call__(self, client, remote_file_name, console):
        def callback_interface(complete, total):
            step = int(round(complete * 100 / total)) - self.progress_value
            self.progress.advance(self.download_tqdm, advance = step)
            self.progress_value = int(round(complete * 100 / total))
        if self.check_file_exist(client, remote_file_name):
            for i in range(0, 10):
                try:
                    response = client.download_file(
                        Bucket = self.bucket_name,
                        Key = remote_file_name,
                        DestFilePath = os.path.join(self.fold_path, remote_file_name),
                        progress_callback = callback_interface)
                    break
                except:
                    print('Error')
        else:
            print('Error')


class RouterMana(object):
    def __init__(self) -> None:
        self.console = Console()
        self.bucket_name = 'router-share-1257105754'
        secret_id = "AKIDjJg05yd5IJ7rQn01rTt68xb2MTMZaMvH"
        secret_key = "CUYOzPfjcIm36zhDVcVAfwELFyWrysZ6"
        region = 'ap-beijing'
        scheme = 'https'
        config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Token=None, Scheme=scheme)
        self.client = CosS3Client(config)

    def push_to_cloud(self, tag:str, router_hf_model_weight_path:str):
        fold_path = os.path.dirname(router_hf_model_weight_path)
        weight = torch.load(router_hf_model_weight_path)
        selected_data = {}
        for k, v in weight.items():
            if 'skip_router' in k:
                selected_data[k] = v
        generated_pt_path = os.path.join(fold_path, tag)
        torch.save(selected_data, generated_pt_path)

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            upfile = UploadFile(progress, self.bucket_name)
            upfile(self.client, generated_pt_path, self.console)
        print(f"Keep this saved tag: {tag}")

    def get_from_cloud(self, tag:str, target_router_hf_model_weight_path:str):
        fold_path = os.path.dirname(target_router_hf_model_weight_path)
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            downfile = DownloadFile(progress, self.bucket_name, fold_path)
            downfile(self.client, tag, self.console)
        target_pt_path = os.path.join(fold_path, tag)
        selected_weight = torch.load(target_pt_path)
        source_weight = torch.load(target_router_hf_model_weight_path)
        new_dict = {}
        for k, v in source_weight.items():
            if 'skip_router' in k:
                new_dict[k] = selected_weight[k]
            else:
                new_dict[k] = v
        torch.save(new_dict, target_router_hf_model_weight_path)
        print(f'Got the cloud skip router to local: {target_router_hf_model_weight_path}')

if __name__ == "__main__":

    # set the weight path.
    router_hf_model_weight_path = 'exps/sr4_qwen2_A/weights/hf_model_0002_0.pt' # NOTE, Must be `0002`, because router storages in here

    # init a manager
    router_mana = RouterMana()

    # upload `router_weights` to cloud, please change the unique tag each time !!!
    router_mana.push_to_cloud(tag = '1234test', router_hf_model_weight_path = router_hf_model_weight_path)

    # Get `router_weights` from cloud, it would cover the local skip router weight, please be tense to operate it
    router_mana.get_from_cloud(tag = '1234test', target_router_hf_model_weight_path = router_hf_model_weight_path)






    
