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

    #V48_PLACES
    #z00
    direct_link = "https://drive.google.com/file/d/1DbAB8ZKRfHXW3r68VefRn1-5HVEVzimM/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z01
    direct_link = "https://drive.google.com/file/d/1DZRXjsTrWnJxzp6ZvilAzOUoj8PvBq0z/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z02
    direct_link = "https://drive.google.com/file/d/1DZhrFApsljZyarQhiBV0Ey7LHLpYnLev/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z03
    direct_link = "https://drive.google.com/file/d/1DZoykRjPK-CbsQzLaH0Jvv_uvwINE_xj/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z04
    direct_link = "https://drive.google.com/file/d/1D_cNOYCjBTSCD-pJlA9stLCezP3zD3Uj/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z05
    direct_link = "https://drive.google.com/file/d/1Da7y6KSSVBpJUSp5jO1e57G6wK99LyjG/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z06
    direct_link = "https://drive.google.com/file/d/1DaarRAfCxDvn_Xlv-YcU796_80tj3pBo/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z07
    direct_link = "https://drive.google.com/file/d/1DavXeQoa7pFvcsF4OZEfL7UdG5s9EoLg/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

