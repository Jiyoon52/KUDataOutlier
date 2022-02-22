import os
import json
from utils.args import outlier_detection_argparser

def main(args):
    # Define Save Path
    args.save_name = f'{args.algorithm}_{args.imputation}_{args.percentile}'
    with open(os.path.join(args.save_dir, f'arg_parser_{args.save_name}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


if __name__ == '__main__':
    parser = outlier_detection_argparser()
    args = parser.parse_args()

    main(args)