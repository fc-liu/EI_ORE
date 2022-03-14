import argparse
# import sys

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_name", default="all",
                        type=str, help="Path of checkpoint")

    parser.add_argument("--max_sentence_length", default=512,
                        type=int, help="Max sentence length in data")

    parser.add_argument("--e11", default="[e11]",
                        type=str, help="start of e1")
    parser.add_argument("--e12", default="[e12]",
                        type=str, help="end of e1")
    parser.add_argument("--e21", default="[e21]",
                        type=str, help="start of e2")
    parser.add_argument("--e22", default="[e22]",
                        type=str, help="end of e2")
    parser.add_argument("--unk", default="[unk]",
                        type=str, help="oov token")
    parser.add_argument("--pad", default="[pad]",
                        type=str, help="pad token")
    parser.add_argument("--cls", default="[CLS]",
                        type=str, help="pad token")
    parser.add_argument("--sep", default="[SEP]",
                        type=str, help="pad token")

    parser.add_argument("--mode", default="train",
                        type=str, help="train or eval")

    parser.add_argument("--bert_model", default="bert-base-uncased",  # default="bert-large-uncased",
                        type=str, help="bert model")
    parser.add_argument("--hidden_size", default=768,
                        type=int, help="Dimensionality of BERT hidden (default: 768)")
    parser.add_argument("--dropout", default=0.6,
                        type=float, help="Dropout probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-6,
                        type=float, help="L2 regularization lambda (default: 1e-5)")
    parser.add_argument("--epochs", default=5,
                        type=int, help="finish pretrain model after this many steps (default: 100)")
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch Size (default: 20)")
    parser.add_argument("--learning_rate", default=1e-5,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--lr_step_size", default=1000,
                        type=int, help="learning schedule per steps")
    parser.add_argument("--decay_rate", default=0.6,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")
    parser.add_argument("--cudas", default=[5], type=int,
                        help="used cudas", nargs='+')
    parser.add_argument("--num_cls", default=10,
                        type=int, help="number of classes")
    args = parser.parse_args()
    return args
    
config = parse_args()