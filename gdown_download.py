import gdown

# a folder
url = "https://drive.google.com/drive/folders/1CMz2flknC81dX3xlKBgNP34NJAQ651DB"
output = "./v26_base/"
gdown.download_folder(url, use_cookies=False)
