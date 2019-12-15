import argparse
import zipfile
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

parser = argparse.ArgumentParser(description='Download data to paperspace')
parser.add_argument('file_id', type=str, help='Link to file download')
parser.add_argument('file_name', type=str, help='Name of downloaded file')

args = parser.parse_args()

gauth = GoogleAuth()
drive = GoogleDrive(gauth)
file = drive.CreateFile({'id': args.file_id})
file.GetContentFile("/storage/data/" + args.file_name)

os.chdir("/storage/data")
zip = zipfile.ZipFile(args.file_name)
zip.extractall()