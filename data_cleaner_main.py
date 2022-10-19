import sys
from optparse import OptionParser
import constants
from loaders import dataset_loader

parser = OptionParser()
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--dataset_version', type=str, default="v21")
def main(argv):
    (opts, args) = parser.parse_args(argv)
    print(opts)

    constants.rgb_dir_ws = "E:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.png"
    constants.rgb_dir_ns = "E:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.png"

    rgb_dir_ws = constants.rgb_dir_ws.format(dataset_version=opts.dataset_version)
    rgb_dir_ns = constants.rgb_dir_ns.format(dataset_version=opts.dataset_version)

    rgb_ws_list = dataset_loader.assemble_img_list(rgb_dir_ws, opts)
    rgb_ns_list = dataset_loader.assemble_img_list(rgb_dir_ns, opts)

    dataset_loader.clean_dataset(rgb_ws_list, rgb_ns_list, 10.0)

if __name__ == "__main__":
    main(sys.argv)