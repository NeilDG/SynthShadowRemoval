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
    # direct_link = "https://drive.google.com/file/d/1luFIjKEE2H81rdnB68ckQz2yri3Ruhfs/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1luQc67YnsVA2U7BFfNdvJjC4sdNsrwBr/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1luVyOlGAcl2L-_lzgmg1ooq2M95EIl2i/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1luWpUYM8D23AKr7Fs3MNjOT-IQ8qk6EF/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1luXacGlQEnXD1yB_ve89jFdn33Z87yXP/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lv1Hlp3IZOqn71Ie8IpLXFAME8_sBPS7/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lvE_maakvZsVm-z_thjRSP7TD3B-Iy7u/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id="+id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lvpxQq7tmx1utyCPkIywyQSpqaybWrHR/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lw2V4H-osFer21JrPsJERJgNxMBtsB4P/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

    # ISTD
    # direct_link = "https://drive.google.com/file/d/1m-GTuXFUiYdhAkehIhzG0jcwdu4tyJVC/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lwF-n8art3wZSwU6ALFpeVK0nUZoYfES/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lwkSojnmN1fHIL24ve4og1-ta2XNwmbL/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lx4Wjap2hxItYHiP6ipb-HXHB5_21GE5/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lxcAhPc3-d0_ImAXSDotMoWNiLe7xv_x/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lyCNDZbPsaIVLXKGSMRlYjf8sc9q8yuY/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lyVK4Mrm8qTxQFLTeFC-bIWi5uyP9qsA/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lyhov-xpMsTLUEwyPSWE3bkr9J_9l9Ln/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1lzQ1_ZGe2t_oXRXXFo99G56K7MEb1Poq/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # #SRD
    # direct_link = "https://drive.google.com/file/d/1m24IabO22ApRRzAA-pH8yPe8zkIKa6QA/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m-Qb4iO4dFMbfu1LvOIevZnXWlNqYsA9/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m-ZHTkoozaIfq3st_3XUILeWhlIhsNZb/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m-pG7UyytNbJ8oY2QpU49L08ULGo8evD/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m0B4hEkhSTMUT1E7sEvtRMQDvDvcsBVy/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m1AC_9OpqXzAlqX5B1iJ1sZAYi8lC4Wj/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m1MTymj3TKoif11bHeyF5bn6nRiGjpfw/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m1S1gXdgbGfMjvYXh1OxSeIPyor0jnZL/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m1_cH853EtITIPiOzxKIQTPJLhEQC034/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)
    #
    # direct_link = "https://drive.google.com/file/d/1m1qFifyal03vVZf99VEu8zfw_z2WTsQH/view?usp=sharing"
    # id = direct_link.split("/d/")[1].split("/")[0]
    # url = "https://drive.google.com/uc?id=" + id
    # gdown.download(url, output=output_dir, use_cookies=False)

    #USR
    url="https://drive.google.com/drive/folders/1m2KBqGJGDl8vITVTp4tnjnrKX1hTEe1k"
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

