import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   
import tensorflow as tf
import numpy as np
from model import Model
import os, time, argparse
import pickle as pkl


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
	parser.add_argument('--log_dir', type=str, default='logs')
	parser.add_argument('--anneal_interval', type=int, default=20)
	parser.add_argument('--initial_lr', type=float, default=0.05)

	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--feature_dim', type=int, default=32)
	parser.add_argument('--mem_state_dim', type=int, default=32)
	parser.add_argument('--final_fc_dim', type=int, default=50)           # 1 10
	args = parser.parse_args()

	dataset = 'mnemosyne'
	t0 = time.time()
	load_path = os.path.join(os.path.dirname(os.getcwd()), 'trunc_feature/%s_trunc_DKVMN.pkl' % (dataset))
	with open(load_path, 'rb') as f:
		q, qa, decay_factor = pkl.load(f)
	print('read data time: ', time.time()-t0)

	args.dataset = dataset
	args.n_items = np.max(q)     
	args.seq_len = q.shape[1]
	args.decay_dim = decay_factor.shape[-1]
	training_step = q.shape[0] // args.batch_size
	args.anneal_interval *= training_step

	targets = ((qa-1)//args.n_items).astype(np.float32)
	real_seq_len = np.sum(q!=0, axis=1)
	real_seq_len_check = np.sum(qa!=0, axis=1)	
	assert real_seq_len.shape[0] == real_seq_len_check.shape[0] 
	assert np.sum(real_seq_len!=real_seq_len_check) == 0	
	print('train_valid_data shape: ', q.shape)

	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)

	run_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	run_config.gpu_options.allow_growth = True
	with tf.Session(config=run_config) as sess:
		MNA = Model(args, sess, name='MNA')
		MNA.train(q, qa, decay_factor, targets, real_seq_len)


if __name__ == "__main__":
	main()
