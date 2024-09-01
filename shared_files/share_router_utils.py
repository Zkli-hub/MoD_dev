from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
import sys
import os
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import time
import shutil

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
    def __init__(self, progress, bucket_name):
        self.progress_value = 0
        self.progress = progress
        self.download_tqdm = self.progress.add_task(description="Progress", total = 100)
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
            if os.path.exists('./m2cos_download') == False:
                os.mkdir('./m2cos_download')
            for i in range(0, 10):
                try:
                    response = client.download_file(
                        Bucket = self.bucket_name,
                        Key = remote_file_name,
                        DestFilePath = './m2cos_download/' + remote_file_name,
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

    def push_to_cloud(self, code_tag, hf_model_weight_path):
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            upfile = UploadFile(progress, self.bucket_name)
            upfile(self.client, weight_path, self.console)

    def get_from_cloud(self, code_tag:str, target_hf_model_weight_path):
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn()) as progress:
            downfile = DownloadFile(progress, self.bucket_name)
            downfile(self.client, target_hf_model_weight_path, self.console)

if __name__ == "__main__":
    router_mana = RouterMana()
    # router_mana.push_to_cloud('1.txt')
    # router_mana.get_from_cloud('', '1.txt')






    