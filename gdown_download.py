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
    direct_link = "https://drive.google.com/file/d/1RJnBHIthwDmpTf9Q0U-LF0r_dgfXQtAf/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z01
    direct_link = "https://drive.google.com/file/d/1RJpR7yVphEgp35HdlkZil0u9SaVoHJYO/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z02
    direct_link = "https://drive.google.com/file/d/1RJrGhcRSE2VIdN7Jztr3WdVPxtu8-Jca/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z03
    direct_link = "https://drive.google.com/file/d/1RJy76bRQqeIgCcH5v6PzSqDR9aXaZFdr/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z04
    direct_link = "https://drive.google.com/file/d/1RL6GJN-HYjw4j0S4VCzb8m0OvHcV5q9j/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z05
    direct_link = "https://drive.google.com/file/d/1RLWeqmhSivId9EDbu2EFnCt6xk5vOSxt/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z06
    direct_link = "https://drive.google.com/file/d/1RLjqBjbpKCTBy_wWFF3halgyvzdyVaLR/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z07
    direct_link = "https://drive.google.com/file/d/1RLx1CvbmyB43ICP-61h-C-n5a_5NGcLi/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

