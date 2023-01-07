import sys

import gdown
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)

def main(argv):
    (opts, args) = parser.parse_args(argv)

    if(opts.server_config == 1):
        output_dir = "/scratch1/scratch2/neil.delgallego/SynthWeather Dataset 10/"
    else:
        output_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/"

    #V51_PLACES
    #z00
    direct_link = "https://drive.google.com/file/d/1HL393RGr4zZOTHn5TugIKnWVkkP9OVXR/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z01
    direct_link = "https://drive.google.com/file/d/1EI0LpWPIfVF3OAdk6gmiQ37sLzNaQmox/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z02
    direct_link = "https://drive.google.com/file/d/1HG8FYNNncQYivC0f8uOK3-LA0Mm7vUf1/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z03
    direct_link = "https://drive.google.com/file/d/1HJ-NWZChi9Ge7XyxmgjMgwjZTqxgkQqt/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z04
    direct_link = "https://drive.google.com/file/d/1HJmzM-fySSJPjE30mXrdFdtryEaW4Y6B/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z05
    direct_link = "https://drive.google.com/file/d/1HKgZQLeJhGoO1-HGz3mxjx0DQ0kR9him/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z06
    direct_link = "https://drive.google.com/file/d/1HKzSkubAz4b5GnZo9i2vI_FawOxwZQLD/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z07
    direct_link = "https://drive.google.com/file/d/1HL2zHW15E5dfFax2sMOcIaLjVRjwpWfD/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

