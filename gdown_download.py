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
    elif (opts.server_config == 3):
        output_dir = "/home/neildelgallego/SynthWeather Dataset 10/"
    else:
        output_dir = "/home/jupyter-neil.delgallego/SynthWeather Dataset 10/"

    #z00
    # direct_link = "https://drive.google.com/file/d/1liR2C3g9M4u1BhrEJSaC4WkkMZdcTC5p/view?usp=share_link"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # #z01
    # direct_link = "https://drive.google.com/file/d/1lgL0PqMWiBr-Trvsi7qWkaL88WUPauwP/view?usp=share_link"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # #z02
    # direct_link = "https://drive.google.com/file/d/1lgqFuTHfXkK-IimryDBUmLfo7s7h9p5w/view?usp=share_link"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)

    #z03
    direct_link = "https://drive.google.com/file/d/1lhswYavVqAfi6YjjKxldQRqPtNgit4LT/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # #z04
    # direct_link = "https://drive.google.com/file/d/1lhuLrwcL_vbifCr5am7je11GLw7p2LUQ/view?usp=share_link"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # #z05
    # direct_link = "https://drive.google.com/file/d/1liIuI1SmOO9nCsQsELp9uX1My6J8KYr1/view?usp=share_link"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # #z06
    # direct_link = "https://drive.google.com/file/d/1liYCjJe1IPV-UZUvC2xaimZMwx9mA2Tm/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)

    #
    # # z07
    # direct_link = "https://drive.google.com/file/d/16zOHdtkoYo0z5NzKbTgg7OCvz5eQo4rC/view?usp=share_link"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

