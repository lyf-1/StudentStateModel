import numpy as np
import tensorflow as tf
import utils
    

class Memory:
    def __init__(self, mem_size, mem_state_dim, name):
        self.name = name
        print("initialize %s" % self.name)
        
        self.mem_size = mem_size
        self.mem_state_dim = mem_state_dim
    
    def cor_weight(self, q_embedded, key_matrix):
        """
            q_embedded: [batch_size, mem_state_dim]
            key_matrix: [mem_size, mem_state_dim]
            correlation_w: [batch_size, mem_size]
        """
        embedding_rst = tf.matmul(q_embedded, tf.transpose(key_matrix))
        correlation_w = tf.nn.softmax(embedding_rst)
        return correlation_w
    
    def read(self, value_matrix, correlation_w):
        """
            value_matrix: [batch_size, mem_size, mem_state_dim]
            correlation_w: [batch_size, mem_size]
            read_content: [batch_size, self.mem_state_dim]
        """
        value_matrix_reshape = tf.reshape(value_matrix, [-1, self.mem_state_dim])
        correlation_w_reshape = tf.reshape(correlation_w, [-1, 1])
        rc = tf.multiply(value_matrix_reshape, correlation_w_reshape)
        read_content = tf.reshape(rc, [-1, self.mem_size, self.mem_state_dim])
        read_content = tf.reduce_sum(read_content, axis=1, keep_dims=False)
        return read_content
    
    def write_decay(self, value_matrix, correlation_w, decay_factor, reuse=False):
        """
            decay_factor: [t, nreps], batch_size*2
        """
        decay_vector = utils.linear(decay_factor, self.mem_state_dim, name=self.name+'/decay_Vector', reuse=reuse)
        decay_signal = tf.sigmoid(decay_vector) 
        decay_signal_reshape = tf.reshape(decay_signal, [-1, 1, self.mem_state_dim]) 
        correlation_w_reshape = tf.reshape(correlation_w, [-1, self.mem_size, 1])
        decay_signal_mul = tf.multiply(decay_signal_reshape, correlation_w_reshape)   # [batch_size, self.mem_size. self.mem_state_dim]
        new_value_matrix = value_matrix * (1 - decay_signal_mul)
        return new_value_matrix

    def write_qa(self, value_matrix, correlation_w, qa_embedded, reuse=False):
        """
            qa_embedded: [batch_size, mem_state_dim]
            value_matrix: [batch_size, mem_size, mem_state_dim]
            correlation_w: [batch_size, mem_size]
            new_value_matrix: [batch_size, mem_size, mem_state_dim]
        """
        erase_vector = utils.linear(qa_embedded, self.mem_state_dim, name=self.name+'/Erase_Vector', reuse=reuse)
        erase_signal = tf.sigmoid(erase_vector)
        add_vector = utils.linear(qa_embedded, self.mem_state_dim, name=self.name+'/Add_Vector', reuse=reuse)
        add_signal = tf.tanh(add_vector)

        erase_signal_reshape = tf.reshape(erase_signal, [-1, 1, self.mem_state_dim])
        correlation_w_reshape = tf.reshape(correlation_w, [-1, self.mem_size, 1])
        erase_mul = tf.multiply(erase_signal_reshape, correlation_w_reshape) # [batch_size, self.mem_size. self.mem_state_dim]
        erase = value_matrix * (1 - erase_mul) 
        add_signal_reshape = tf.reshape(add_signal, [-1, 1, self.mem_state_dim])
        add_mul = tf.multiply(add_signal_reshape, correlation_w_reshape)

        new_value_matrix = erase + add_mul
        return new_value_matrix


class DKVMN:
    def __init__(self, mem_size, key_mem_state_dim, value_mem_state_dim, 
                 init_key_mem, init_value_mem, name='DKVMN'):
        self.name = name
        self.mem_size = mem_size
        self.key_mem_state_dim = key_mem_state_dim
        self.value_mem_state_dim = value_mem_state_dim

        self.key = Memory(self.mem_size, self.key_mem_state_dim, name=self.name+'_key_matrix')
        self.value = Memory(self.mem_size, self.value_mem_state_dim, name=self.name+'_value_matrix')
        
        self.key_mem = init_key_mem
        self.value_mem = init_value_mem

    def attention(self, q_embedded):
        return self.key.cor_weight(q_embedded, self.key_mem)

    def read(self, correlation_w):
        return self.value.read(self.value_mem, correlation_w)
    
    def write_qa(self, correlation_w, qa_embedded, reuse):
        self.value_mem = self.value.write_qa(self.value_mem, correlation_w, qa_embedded, reuse=reuse) 
        return self.value_mem

    def write_decay(self, correlation_w, decay_factor, reuse):
        self.value_mem = self.value.write_decay(self.value_mem, correlation_w, decay_factor, reuse=reuse)
        return self.value_mem

        






