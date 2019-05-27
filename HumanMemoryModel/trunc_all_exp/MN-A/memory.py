import numpy as np
import tensorflow as tf
import utils
    

class Memory:
    def __init__(self, mem_size, mem_state_dim, name):
        self.name = name
        # print("initialize %s" % self.name)
        
        self.mem_size = mem_size
        self.mem_state_dim = mem_state_dim
    
    def read(self, score, key_matrix, reuse=False):
        """
            score: [batch_size, mem_size]
            key_matrix: [batch_size, mem_size, mem_state_dim]
        """
        key_matrix_reshape = tf.reshape(key_matrix, [-1, self.mem_state_dim])
        score_reshape = tf.reshape(score, [-1, 1])
        rc = tf.multiply(key_matrix_reshape, score_reshape)
        read_content = tf.reshape(rc, [-1, self.mem_size, self.mem_state_dim])
        read_content = tf.reduce_sum(read_content, axis=1, keep_dims=False)
        return read_content
    
    def write_decay(self, key_matrix, q, decay_factor, reuse=False):
        """
            decay_factor: [tlast, nreps], batch_size*2
        """
        mask = tf.one_hot(q, depth=self.mem_size)      
        mask = tf.reshape(mask, [-1, self.mem_size, 1])
        decay_vector = utils.linear(decay_factor, self.mem_state_dim, name=self.name+'/time_Vector', reuse=reuse)
        decay_signal = tf.sigmoid(decay_vector)   # [batch_size, key_mem_state_dim]
        decay_signal_reshape = tf.reshape(decay_signal, [-1, 1, self.mem_state_dim])
        decay_update = tf.multiply(decay_signal_reshape, mask) 
        new_key_matrix = key_matrix * decay_update
        return new_key_matrix

    def write_qa(self, key_matrix, q, qa_embedded, reuse=False):
        """
            key_matrix: [batch_size, key_mem_size, key_mem_state_dim]
            qa_embedded: [batch_size, key_mem_state_dim]
            q: [batch_size,]
            t: [batch_size, 1]
            new_key_matrix: same as key matrix
        """
        mask = tf.one_hot(q, depth=self.mem_size)
        mask = tf.reshape(mask, [-1, self.mem_size, 1])
        erase_vector = utils.linear(qa_embedded, self.mem_state_dim, name=self.name+'/Erase_vector', reuse=reuse)
        erase_signal = tf.sigmoid(erase_vector)
        add_vector = utils.linear(qa_embedded, self.mem_state_dim, name=self.name+'/Add_Vector', reuse=reuse)
        add_signal = tf.tanh(add_vector)   # [batch_size, key_mem_state_dim] 

        erase_signal_reshape = tf.reshape(erase_signal, [-1, 1, self.mem_state_dim])
        erase_mul = tf.multiply(erase_signal_reshape, mask)      # [batch_size, self.mem_size. self.mem_state_dim]
        erase = key_matrix * (1 - erase_mul)
        add_signal_reshape = tf.reshape(add_signal, [-1, 1, self.mem_state_dim])
        add_mul = tf.multiply(add_signal_reshape, mask)

        new_key_matrix = erase + add_mul        
        return new_key_matrix


class MN:
    def __init__(self, mem_size, mem_state_dim, init_mem, name='MN'):
        self.name = name
        self.mem_size = mem_size
        self.mem_state_dim = mem_state_dim

        self.MemModel = Memory(self.mem_size, self.mem_state_dim, name=self.name+'_matrix')
        self.mem = init_mem

    def read(self, score, reuse):
        """
            score: [batch_size, n_items+1]
            return: the value of one key memory slot
                    shape: [batch_size, mem_state_dim]
        """
        return self.MemModel.read(score, self.mem, reuse=reuse)
    
    def write_decay(self, q, decay_factor, reuse):
        self.mem = self.MemModel.write_decay(self.mem, q, decay_factor, reuse=reuse)
        return self.mem

    def write_qa(self, q, qa_embedded, reuse):
        self.mem = self.MemModel.write_qa(self.mem, q, qa_embedded, reuse=reuse)
        return self.mem

        






