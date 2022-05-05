import argparse
import logging
import numpy as np
import pandas as pd
from tdc import Evaluator
from tdc import Oracle


def read_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--generated_path', required=True)
    add_arg('--train_path', required=True)
    add_arg('--name', required=True)
    return parser.parse_args()


def main():
    args = read_args()
    gen_data = pd.read_csv(args.generated_path)['SMILES'].to_list()[-1000:]
    train_data = pd.read_csv(args.train_path, header=None)[1].to_list()

    logging.basicConfig(filename=str(args.name) + "_tdcmetrics.txt", level=logging.INFO)

    div_evaluator = Evaluator(name = 'Diversity')
    kl_evaluator = Evaluator(name = 'KL_Divergence')
    nov_evaluator = Evaluator(name = 'Novelty')
    val_evaluator = Evaluator(name = 'Validity')
    uniq_evaluator = Evaluator(name = 'Uniqueness')
    sa_oracle = Oracle(name = 'SA')
    qed_oracle = Oracle(name = 'QED')
    fcd_evaluator = Evaluator(name = 'FCD_Distance')

    logging.info("")
    logging.info("Diversity: " + str(div_evaluator(gen_data)))
    logging.info("Validity: " + str(val_evaluator(gen_data)))
    logging.info("Uniqueness: " + str(uniq_evaluator(gen_data)))
    logging.info("QED: " + str(np.mean(qed_oracle(gen_data))))
    logging.info("SA: " + str(np.mean(sa_oracle(gen_data))))
    logging.info("FCD: " + str(fcd_evaluator(gen_data, train_data)))
    logging.info("KL_Divergence: " + str(kl_evaluator(gen_data, train_data)))
    logging.info("Novelty: " + str(nov_evaluator(gen_data, train_data)))

if __name__ == "__main__":
    main()
