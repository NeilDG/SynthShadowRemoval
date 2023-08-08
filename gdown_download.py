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

    # ISTD train dataset
    # direct_link = "https://drive.google.com/file/d/1liYCjJe1IPV-UZUvC2xaimZMwx9mA2Tm/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # #SRD Train dataset
    # direct_link = "https://drive.google.com/file/d/1lipMqaY7NfNevhq5uC_S06Rf4swUK5RX/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

    # v88_istd
    # direct_link = "https://drive.google.com/file/d/1BPNQOn6ss0eOLYMWk3iEeQg_n6vE3JuJ/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

    # v89_istd
    # direct_link = "https://drive.google.com/file/d/1BT4BwvEYTjsz_wCG0CfHY5upr_wf19mY/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False, quiet=False)

    # v90_istd
    direct_link = "https://drive.google.com/file/d/1BUWt-uS-LkwE86ICnbQncdbe6kG2wZwR/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False, quiet=False)

    # v91_istd
    direct_link = "https://drive.google.com/file/d/1BVRFiJxgyxOZaCyAjdjux1yzmj-iHyF9/view?usp=drive_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False, quiet=False)

    #v_istd+srd
    # direct_link = "https://drive.google.com/file/d/1mmI14uOtZzXzVX3P2AwRmdfkmNvJCUQe/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False, quiet=False)

    # url = "https://drive.google.com/drive/folders/1mhKmxwDODP4aCccG39FyQLlR0QVBvNA2?usp=sharing"
    # gdown.download_folder(url, output=output_dir, use_cookies=False)

    # url = "https://drive.google.com/drive/folders/1mhLqCankhs2i2sH2MGHea9kUCAbsiDy_?usp=sharing"
    # gdown.download_folder(url, output=output_dir, use_cookies=False)

    #v66_usr dataset
    # url = "https://drive.google.com/drive/folders/1m2KBqGJGDl8vITVTp4tnjnrKX1hTEe1k?usp=sharing"
    # gdown.download_folder(url, output=output_dir, use_cookies=False)
    #

if __name__ == "__main__":
    main(sys.argv)

