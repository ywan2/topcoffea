import argparse
import json
import datetime
from topcoffea.modules.YieldTools import YieldTools
import topcoffea.modules.MakeLatexTable as mlt

# This script takes a pkl file, finds the yields in the analysis categories, saves the yields to a json
#   - If you do not specify a pkl file path, will default to "hists/plotsTopEFT.pkl.gz"
#   - Example usage: python get_yield_json.py -f histos/plotsTopEFT.pkl.gz

def main():

    yt = YieldTools()

    timestamp_tag = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--pkl-file-path", default="histos/plotsTopEFT.pkl.gz", help = "The path to the pkl file")
    parser.add_argument("-y", "--year", default="2017", help = "The year of the sample")
    parser.add_argument("-t", "--tag", default="Sample", help = "A string to describe the pkl file")
    parser.add_argument("-n", "--json_name", default="yields", help = "Name of the json file to save")
    parser.add_argument("-q", "--quiet", action="store_true", help = "Do not print out anything")
    args = parser.parse_args()

    # Get the histograms, and put the yields into a dict
    hin_dict = yt.get_hist_from_pkl(args.pkl_file_path)
    yld_dict = yt.get_yld_dict(hin_dict,args.year)

    # Print info about the file
    if not args.quiet:
        yt.print_hist_info(args.pkl_file_path)
        yt.print_yld_dicts(yld_dict,args.tag)
        mlt.print_latex_yield_table(yld_dict,key_order=yt.PROC_MAP.keys(),subkey_order=yt.CAT_LST,tag=args.tag,print_begin_info=True,print_end_info=True)

    # Save to a json
    out_json_name = args.json_name
    if args.json_name == parser.get_default("json_name"):
        out_json_name = out_json_name + "_" + timestamp_tag
    with open(out_json_name+".json", "w") as out_file:
        json.dump(yld_dict, out_file, indent=4)
    print(f"Saved json file: {out_json_name}.json\n")
    
if __name__ == "__main__":
    main()
