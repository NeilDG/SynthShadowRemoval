import gdown

# a folder
# url = "https://drive.google.com/drive/folders/1CMz2flknC81dX3xlKBgNP34NJAQ651DB"
# output = "./v26_base/"
# gdown.download_folder(url, output=output, use_cookies=False)

url = "https://drive.google.com/uc?id=1NLYQA3vn_-tB-YTxG_wp7s5O1T0O9tAm"
output = "./v26_base/"
gdown.download(url, use_cookies=False, output=output)

# #separate files
# #z04
# url = "https://drive.google.com/uc?id=1CdudMs-yb75Zxa42DUQ_MqqU9HN0utwP"
# output = "./v26_base/"
# gdown.download(url, output=output)
#
# #z05
# url = "https://drive.google.com/uc?id=1Cfqv0U6vPiHDaeBWWlOwNucY6A3iX0wN"
# output = "./v26_base/"
# gdown.download(url, output=output)
#
# #z06
# url = "https://drive.google.com/uc?id=1CjN64ivdfhtEBy37HvhXEvMa4052ZXkk"
# output = "./v26_base/"
# gdown.download(url, output=output)