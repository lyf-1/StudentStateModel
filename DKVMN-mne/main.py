import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   
import tensorflow as tf
import numpy as np
from model import Model
import os, time, argparse
import pickle as pkl


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'n', 'f', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Not expected boolean type')


def split_data(datalist, rate=0.8):
    splitpoint = int(datalist[0].shape[0] * 0.8)
    splitdata = []
    for data in datalist:
        train = data[:splitpoint, :]
        valid = data[splitpoint:, :]
        splitdata.append(train)
        splitdata.append(valid)
    return splitdata


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_epochs', type=int, default=500)
	parser.add_argument('--train', type=str2bool, default='t')
	parser.add_argument('--init_from', type=str2bool, default='t')
	parser.add_argument('--show', type=str2bool, default='f')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
	parser.add_argument('--log_dir', type=str, default='logs_nreps')
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--anneal_interval', type=int, default=20)
	parser.add_argument('--maxgradnorm', type=float, default=50.0)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--initial_lr', type=float, default=0.05)

	parser.add_argument('--batch_size', type=int, default=17)
	parser.add_argument('--mem_size', type=int, default=50)
	parser.add_argument('--key_mem_state_dim', type=int, default=50)
	parser.add_argument('--value_mem_state_dim', type=int, default=200)
	parser.add_argument('--final_fc_dim', type=int, default=50)
	args = parser.parse_args()
	
	dataset = 500
	args.dataset = dataset
	if dataset == 500:
		load_path = 'feature/mnemosyne_2126_40699_500.pkl'
	elif dataset == 1000:
		load_path = 'feature/mnemosyne_2590_72272_1000.pkl'
	elif dataset == 2174:
		load_path = 'feature/mnemosyne_2742_88892_2174.pkl'
	else:
		print('no dataset')
		exit()
	q, qa, tlast, nreps = pkl.load(open(load_path, 'rb'))
	args.n_items = int(load_path.split('_')[2])     
	args.seq_len = q.shape[1]
	ans = ((qa-1)//args.n_items).astype(np.float32)
	datalist_ = [q, qa, ans, tlast, nreps]
	train_q, valid_q, train_qa, valid_qa, train_targets, valid_targets, \
			 train_t, valid_t, train_nreps, valid_nreps = split_data(datalist_, 0.8)
	print('q_data shape(train/valid): ', train_q.shape, valid_q.shape)
	training_step = train_q.shape[0] // args.batch_size
	args.anneal_interval *= training_step

	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	run_config.gpu_options.allow_growth = True
	with tf.Session(config=run_config) as sess:
		dkvmn = Model(args, sess, name='DKVMN')
		if args.train:
			dkvmn.train(train_q, train_qa, train_t, train_nreps, train_targets, valid_q, valid_qa, valid_t, valid_nreps, valid_targets)
			# pass
		else:
			pass		


if __name__ == "__main__":
	main()

