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
    direct_link = "https://drive.google.com/file/d/1Deq_EoLmv1p45WfcCM57RuXMPYtb0YET/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z01
    direct_link = "https://drive.google.com/file/d/1Dc1pQXuHsVaA4VP7aOpzJ4iMgt4fvkM0/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z02
    direct_link = "https://drive.google.com/file/d/1DcCHAorozGF1XLHcOWzYNPBEfpe0BHSL/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z03
    direct_link = "https://drive.google.com/file/d/1DcicH9ewQ0yxa0H0GlA2TWQI-3f6veum/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z04
    direct_link = "https://drive.google.com/file/d/1Dd28m9SGlP0mS5SpU63RM2yu3iXnedmV/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z05
    direct_link = "https://drive.google.com/file/d/1De8xF5r0tV9ZqUgYnQI3mqbTu5kXToEt/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z06
    direct_link = "https://drive.google.com/file/d/1DeUMBueAz6Ubw4Mf1jA1iH34HO9RBBMT/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z07
    direct_link = "https://drive.google.com/file/d/1DellHV0sr4z3IA4qVMV24H6ShBMRzL1_/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

