import argparse
import pandas as pd
import numpy as np


def filter_data(input_file, output_file):
    df = pd.read_csv(input_file, sep=r"\s(.*)", engine='python', header=None)
    df.drop(df.columns[2], axis=1, inplace=True)
    df.columns = ["label", "product_name"]
    filtered_labels = df['label'].value_counts()[lambda x: x > min_prod_count]
    df = df[df.label.isin(filtered_labels.keys())].reset_index(drop=True)
    np.savetxt(output_file, df.values, fmt='%s')

input_file = "/workspace/datasets/fasttext/labeled_products.txt"
output_file = "/workspace/datasets/fasttext/pruned_labeled_products.txt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    general = parser.add_argument_group("general")
    general.add_argument("--input", default=input_file,  help="file containing product data")
    general.add_argument("--output", default="/workspace/datasets/fasttext/pruned_labeled_products.txt", help="file to output pruned product data")
    general.add_argument("--min_prod_count", default=500, help="eg. min_prod_count= 500; entries associated with a label assigned to less than 500 products would get filtered")

    args = parser.parse_args()
    if args.input:
        input_file = args.input
    if args.output:
        output_file = args.output
    min_prod_count = args.min_prod_count
    
    filter_data(input_file, output_file)
    print ("data filtered and saved to file:", output_file)
      
