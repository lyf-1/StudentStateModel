import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
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


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train', type=str2bool, default='f')
	parser.add_argument('--init_from', type=str2bool, default='t')
	parser.add_argument('--show', type=str2bool, default='f')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
	parser.add_argument('--log_dir', type=str, default='logs')
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--anneal_interval', type=int, default=20)
	parser.add_argument('--maxgradnorm', type=float, default=50.0)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--initial_lr', type=float, default=0.05)

	# add decay factor
	parser.add_argument('--proj_len', type=int, default=32)
	parser.add_argument('--decay_factor_in_input', type=str, default='noTime')   # timeMask, timeJoint, noTime, timeConcate
	parser.add_argument('--decay_factor_in_recurrent', type=int, default=0)   # 0 or 1

	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--mem_size', type=int, default=5)
	parser.add_argument('--key_mem_state_dim', type=int, default=64)
	parser.add_argument('--value_mem_state_dim', type=int, default=64)
	parser.add_argument('--final_fc_dim', type=int, default=50)
	args = parser.parse_args()

	dataset = 'efc'
	t0 = time.time()
	load_path = os.path.join(os.path.dirname(os.getcwd()), 'trunc_feature/%s_trunc_DKVMN.pkl' % (dataset))
	with open(load_path, 'rb') as f:
		## decay factor = [tlast, nreps, success, fail]
		## different decay feature combination
		q, qa, decay_factor = pkl.load(f) 
		# decay_factor = np.concatenate([decay_factor[:, :, 0:1],decay_factor[:, :, 2:]], axis=2) 
		# decay_factor = decay_factor[:, :, 1:]

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
		dkvmn = Model(args, sess, name='DKVMN')
		if args.train:
			dkvmn.train(q, qa, decay_factor, targets, real_seq_len)
		else:
			with tf.Session(config=run_config) as sess: 
				saver = tf.train.Saver()
				saver.restore(sess, 'checkpoint/1558353947/DKVMN-18.ckpt')

				batch_q = q[0, :]
				batch_qa = qa[0, :]
				batch_decay_factor = decay_factor[0, :, :]
				batch_target = targets[0, :]
				batch_real_seq_len = real_seq_len[0]

				feed_dict = {dkvmn.q_data:batch_q, dkvmn.qa_data:batch_qa, dkvmn.target:batch_target, dkvmn.decay_factor:batch_decay_factor, dkvmn.real_seq_len:batch_real_seq_len}
				a, b = dkvmn.sess.run([dkvmn.train_op, dkvmn.valid_target, dkvmn.valid_pred], feed_dict=feed_dict)
				print(a)
				# with open('dkvmnatten.pkl', 'wb') as f:
				# 	pkl.dump(a, f, protocol=4)


if __name__ == "__main__":
	main()

