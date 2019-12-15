# Import Google libraries
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
import googleapiclient.errors

# Import general libraries
from argparse import ArgumentParser
from os import chdir, listdir, stat, path
from sys import exit
import ast


def authenticate():
    """
		Authenticate to Google API
	"""

    gauth = GoogleAuth()

    return GoogleDrive(gauth)


def get_folder_id(drive, parent_folder_id, folder_name):
    """
		Check if destination folder exists and return it's ID
	"""

    # Auto-iterate through all files in the parent folder.
    file_list = GoogleDriveFileList()
    try:
        file_list = drive.ListFile(
			{'q': "'{0}' in parents and trashed=false".format(parent_folder_id)}
		).GetList()
    # Exit if the parent folder doesn't exist
    except googleapiclient.errors.HttpError as err:
        # Parse error message
        message = ast.literal_eval(err.content)['error']['message']
        if message == 'File not found: ':
            print(message + folder_name)
            exit(1)
        # Exit with stacktrace in case of other error
        else:
            raise

    # Find the the destination folder in the parent folder's files
    for file1 in file_list:
        if file1['title'] == folder_name:
            print('title: %s, id: %s' % (file1['title'], file1['id']))
            return file1['id']


def create_folder(drive, folder_name, parent_folder_id):
    """
		Create folder on Google Drive
	"""

    folder_metadata = {
        'title': folder_name,
        # Define the file type as folder
        'mimeType': 'application/vnd.google-apps.folder',
		# ID of the parent folder
		'parents': [{"kind": "drive#fileLink", "id": parent_folder_id}]
    }

    folder = drive.CreateFile(folder_metadata)
    folder.Upload()

    # Return folder informations
    print('title: %s, id: %s' % (folder['title'], folder['id']))
    return folder['id']


def upload_files(drive, folder_id, src_folder_name):
    """
		Upload files in the local folder to Google Drive
	"""

    # Enter the source folder
    try:
        chdir(src_folder_name)
    # Print error if source folder doesn't exist
    except OSError:
        print(src_folder_name + 'is missing')
    # Auto-iterate through all files in the folder.
    for file1 in listdir('.'):
        # Check the file's size
        statinfo = stat(file1)
        if statinfo.st_size > 0:
            print('uploading ' + file1)
            # Upload file to folder.
            f = drive.CreateFile(
                {"parents": [{"kind": "drive#fileLink", "id": folder_id}]})
            f.SetContentFile(file1)
            f.Upload()
        # Skip the file if it's empty
        else:
            print('file {0} is empty'.format(file1))

def upload_file(drive, folder_id, file_path):
    # Check the file's size
    print('uploading ' + file_path)
    # Upload file to folder.
    f = drive.CreateFile(
        {"parents": [{"kind": "drive#fileLink", "id": folder_id}]})
    f.SetContentFile(file_path)
    f.Upload()

    return f['id']

def share_file_id(drive, file_id):
    file = drive.CreateFile({'id': file_id})
    file.InsertPermission({
        'type': 'anyone',
        'value': 'anyone',
        'role': 'reader'})

    file.FetchMetadata()

    return file['alternateLink']

def handle(src_folder_name, src_file_name, dst_folder_name, parent_folder_name="root"):
    # Authenticate to Google API
    drive = authenticate()
    # Get parent folder ID
    parent_folder_id = 'root'
    if(parent_folder_name != 'root'):
        parent_folder_id = get_folder_id(drive, 'root', parent_folder_name)

    # Get destination folder ID
    folder_id = get_folder_id(drive, parent_folder_id, dst_folder_name)
    # Create the folder if it doesn't exists
    if not folder_id:
        print('creating folder ' + dst_folder_name)
        folder_id = create_folder(drive, dst_folder_name, parent_folder_id)
    else:
        print('folder {0} already exists'.format(dst_folder_name))

    # Upload the files
    file_path = path.join(src_folder_name, src_file_name)
    file_id = upload_file(drive, folder_id, file_path)
    share_file_id(drive, file_id)
    return file_id