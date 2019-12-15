#You need https://github.com/labbots/google-drive-upload installed
#Client Id: 748733995773-51r0ourip0jt1n38kffrdb98rkahksc7.apps.googleusercontent.com
#Client secret: 2FWR3Gr7f98oRYPdp0g6RnFa

import os
import argparse
import tools.google_drive_upload
import zipfile

upload_dir = "/tmp/metos3d/upload"
path = os.getcwd()
print(path)
parser = argparse.ArgumentParser(description='Upload data to paperspace')
parser.add_argument('local_test_data_dir', type=str, help='Dir where test data is stored locally')
parser.add_argument('google_drive_test_data_dir', type=str, help='Dir where test data should be stored in google drive')
parser.add_argument('project_id', type=str, help='Paperspace project id')

args = parser.parse_args()


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            # Write file flat instead
            ziph.write(os.path.join(root, file), file)

os.chdir(upload_dir)

print("Zipping files")
file_name = f'{args.google_drive_test_data_dir}.zip'
zipf = zipfile.ZipFile(file_name, 'w', zipfile.ZIP_DEFLATED)
zipdir(args.local_test_data_dir, zipf)
zipf.close()

os.chdir(path)
file_id = tools.google_drive_upload.handle(upload_dir, file_name, args.google_drive_test_data_dir)

print("Finished upload")
print(file_id)

print("Downloading data")
os.chdir(path)
gradientCmd = f'gradient jobs create  --name transportnn-data-upload --machineType GPU+ --projectId {args.project_id} --container nziermann/transportnn --command "python download_data.py {file_id} {file_name}"'
print(f'Running: {gradientCmd}')
os.system(gradientCmd)
print("Finished")