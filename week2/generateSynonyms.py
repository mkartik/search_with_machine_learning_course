import argparse
import fasttext
import csv

fasttext.FastText.eprint = lambda x: None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    general = parser.add_argument_group("general")
    general.add_argument("--input", default="/workspace/datasets/fasttext/top_words.txt", nargs ='+', help="file containing words")
    general.add_argument("--model", default="/workspace/model/fasttext_norm_title_model_epoch25_minCount20.bin", nargs ='+', help="file containing pretrained fasttext model")
    general.add_argument("--threshold", default=0.75, nargs ='+', help="threshold for synonyms obtained from fasttext model")
    general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", nargs ='+', help="file to output synonyms for words in input file")

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    threshold = args.threshold

    print ("fasttext model loaded")
    model = fasttext.load_model(args.model)

    with open(output_file, 'w') as outfile:
        writer = csv.writer(outfile)
        with open(input_file, 'r') as f:
            for word in f:
                word = word.rstrip()
                nearest_neighbors = model.get_nearest_neighbors(word)
                line = [word]
                line.extend([synonym[1] for synonym in nearest_neighbors if synonym[0]> threshold ])
                writer.writerow(line)  
    print ("Synonyms written to ouput file: %s" % output_file)
