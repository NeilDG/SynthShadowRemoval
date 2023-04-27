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

    # z00
    direct_link = "https://drive.google.com/file/d/1lle91xoeAzKMOMqLfc9fJ2Hm4nLUakbU/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z01
    direct_link = "https://drive.google.com/file/d/1llUOPHG2XTAh3XktEAAhDxR0fv1Bi1Gw/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z00
    direct_link = "https://drive.google.com/file/d/1ln8CHsnQc72hdJgD9cX-j2m4jRjtXFF9/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z01
    direct_link = "https://drive.google.com/file/d/1lmJ5jUsKwnLdA_Lr9zR7hO3235xS6dxt/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z02
    direct_link = "https://drive.google.com/file/d/1lo1WuC4iV0KYhW3a-3byBmDWDLKfjlnR/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z03
    direct_link = "https://drive.google.com/file/d/1lnrfRxvt7GkvhOQp3JisyJrGgefjUm87/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z04
    direct_link = "https://drive.google.com/file/d/1lpfsWP-7idWeAOCmoR8OxXJ40vlV_XM1/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z05
    direct_link = "https://drive.google.com/file/d/1lp1EiVpRLYC2C_CaQlQWxLWlMQOIlEDM/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z06
    direct_link = ""
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z07
    direct_link = "https://drive.google.com/file/d/1ltgydWIXWp9Jj5QJ9e1ft-GYZM5Yv908/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z08
    direct_link = "https://drive.google.com/file/d/1ltuIehnvXhAEOXAmRkx9dRbMe80-1_-3/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z09
    direct_link = "https://drive.google.com/file/d/1lsw2NQpLgZyu3gwJjzgm__QNoeLY9Up3/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z10
    direct_link = "https://drive.google.com/file/d/1lsHTJd_cUbcqXHDHaBXNJeuqz_2_9ufe/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z11
    direct_link = "https://drive.google.com/file/d/1lrf_31FLR5KmQ_oLt4firehvqmoriHtc/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z12
    direct_link = "https://drive.google.com/file/d/1lrM7Hse0jJWrUb7O5Ex-OId_NnYJIoT4/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z13
    direct_link = "https://drive.google.com/file/d/1lterrCOKUAyAzdfJlNJPMuWRCQm3nD0E/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z14
    direct_link = "https://drive.google.com/file/d/1ltOe7cM1f51cfPFOI7BnRYySUCC57AJJ/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

