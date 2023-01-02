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
    direct_link = "https://drive.google.com/file/d/1DXF8EvM03LrB_j7H7DRgaRrMI2vV9KPV/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z01
    direct_link = "https://drive.google.com/file/d/1DXFBdmWgbJ0mZw-bHFz2G4v-OIGRVCML/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z02
    direct_link = "https://drive.google.com/file/d/1DXNngypIaxnP6bgaAq8FxCefJx3Fj8cd/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z03
    direct_link = "https://drive.google.com/file/d/1DXTQIedIt_76UOstyd3kw4JpdokZUucr/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z04
    direct_link = "https://drive.google.com/file/d/1DXn6nxFpFSEKgFhyJ8qC56bqf6Si8Pe-/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z05
    direct_link = "https://drive.google.com/file/d/1DYpAk90uDCGfVaft8owkFQLBQM-GFO16/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z06
    direct_link = "https://drive.google.com/file/d/1DZ1wxddU6E49NlohPNjVcDQbDfVDklu1/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    #z07
    # direct_link = ""
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

