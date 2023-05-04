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
    # direct_link = "https://drive.google.com/file/d/1liR2C3g9M4u1BhrEJSaC4WkkMZdcTC5p/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1liIuI1SmOO9nCsQsELp9uX1My6J8KYr1/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lgL0PqMWiBr-Trvsi7qWkaL88WUPauwP/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lgqFuTHfXkK-IimryDBUmLfo7s7h9p5w/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lhswYavVqAfi6YjjKxldQRqPtNgit4LT/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lhuLrwcL_vbifCr5am7je11GLw7p2LUQ/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1lw71619_nFyMXstZegYBCOgoTdRJoDoq/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1luFIjKEE2H81rdnB68ckQz2yri3Ruhfs/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1luQc67YnsVA2U7BFfNdvJjC4sdNsrwBr/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1luVyOlGAcl2L-_lzgmg1ooq2M95EIl2i/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1luWpUYM8D23AKr7Fs3MNjOT-IQ8qk6EF/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1luXacGlQEnXD1yB_ve89jFdn33Z87yXP/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1lv1Hlp3IZOqn71Ie8IpLXFAME8_sBPS7/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1lvE_maakvZsVm-z_thjRSP7TD3B-Iy7u/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1lvpxQq7tmx1utyCPkIywyQSpqaybWrHR/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    direct_link = "https://drive.google.com/file/d/1lw2V4H-osFer21JrPsJERJgNxMBtsB4P/view?usp=sharing"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # direct_link = ""
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = ""
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

    # SRD Train
    # direct_link = "https://drive.google.com/file/d/1lipMqaY7NfNevhq5uC_S06Rf4swUK5RX/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

    # # z07
    # direct_link = "https://drive.google.com/file/d/1ltgydWIXWp9Jj5QJ9e1ft-GYZM5Yv908/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # # z08
    # direct_link = "https://drive.google.com/file/d/1ltuIehnvXhAEOXAmRkx9dRbMe80-1_-3/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # # z09
    # direct_link = "https://drive.google.com/file/d/1lsw2NQpLgZyu3gwJjzgm__QNoeLY9Up3/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # # z10
    # direct_link = "https://drive.google.com/file/d/1lsHTJd_cUbcqXHDHaBXNJeuqz_2_9ufe/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # # z11
    # direct_link = "https://drive.google.com/file/d/1lrf_31FLR5KmQ_oLt4firehvqmoriHtc/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # # z12
    # direct_link = "https://drive.google.com/file/d/1lrM7Hse0jJWrUb7O5Ex-OId_NnYJIoT4/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # # z13
    # direct_link = "https://drive.google.com/file/d/1lterrCOKUAyAzdfJlNJPMuWRCQm3nD0E/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # # z14
    # direct_link = "https://drive.google.com/file/d/1ltOe7cM1f51cfPFOI7BnRYySUCC57AJJ/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

