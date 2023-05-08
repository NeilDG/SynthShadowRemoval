import sys

import gdown
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)

def main(argv):
    (opts, args) = parser.parse_args(argv)

    if (opts.server_config == 0):
        output_dir = "/scratch3/neil.delgallego/SynthWeather Dataset 10/"
    elif (opts.server_config == 4):
        output_dir = "D:/NeilDG/Datasets/SynthWeather Dataset 10/"
    elif (opts.server_config == 5):
        output_dir = "/home/neildelgallego/SynthWeather Dataset 10/"
    else:
        output_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/"

    # PLACES
    # direct_link = "https://drive.google.com/file/d/1lw71619_nFyMXstZegYBCOgoTdRJoDoq/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #

    # PLACES
    url = "https://drive.google.com/drive/folders/1lu937F5oUq2ZpBHTOhXc1cdhRmIsQaoH?usp=sharing"
    gdown.download_folder(url, output=output_dir, use_cookies=False)

    # ISTD
    url = "https://drive.google.com/drive/folders/1lw73Dg3xQFbAuGUSFRnzXa89B2fv3-Bo?usp=sharing"
    gdown.download_folder(url, output=output_dir, use_cookies=False)

    # SRD
    url = "https://drive.google.com/drive/folders/1m-PbdSJPuGs_kWh-H0wMjjEvuvmXPg4g?usp=sharing"
    gdown.download_folder(url, output=output_dir, use_cookies=False)

    #USR
    url="https://drive.google.com/drive/folders/1m2KBqGJGDl8vITVTp4tnjnrKX1hTEe1k"
    gdown.download_folder(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

