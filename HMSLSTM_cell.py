#reference: https://github.com/n-s-f/hierarchical-rnn/tree/master/hmlstm
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
import tensorflow as tf
import collections


HMLSTMStateTuple=collections.namedtuple('HMLSTMStateTuple',['c','h','z'])

class HMSLSTM_cell(core_rnn_cell.RNNCell):
    def __init__(self,hstate_size,h_below_size,h_above_size,batch_size,keep_p,reuse):
        super().__init__(_reuse=reuse)
        self.hstate_size=hstate_size
        self.h_below_size=h_below_size
        self.h_above_size=h_above_size
        self.batch_size=batch_size
        self.keep_p=keep_p




    def zero_state(self):
        return HMLSTMStateTuple(c=tf.zeros([self.batch_size,self.hstate_size]),h=tf.zeros([self.batch_size,self.hstate_size]),z=tf.zeros([self.batch_size,1]))

    @property
    def state_size(self):
        return (self.hstate_size_size,self.hstate_size,1)

    @property
    def output_size(self):
        return self.hstate_size_size+1


    def call(self,input,states):
        h=states.h
        c=states.c
        z=states.z
        ha,hb,z_b=tf.split(input,[self.h_above_size,self.h_below_size,1],1)
        s_rec=h
        s_td=z*ha
        s_bu=z_b*hb
        bias_init = tf.constant_initializer(0, dtype=tf.float32)
        concat=core_rnn_cell._linear([s_rec,s_td,s_bu],4*self.hstate_size+1,bias=True,bias_initializer=bias_init)  #[B,4d+1]  ,d is the state_size
        pre_f,pre_i,pre_o,pre_g,pre_z_next = tf.split(concat, [self.hstate_size, self.hstate_size,self.hstate_size,self.hstate_size, 1], 1)

        i = tf.sigmoid(pre_i)  # [B, h_l]
        g = tf.tanh(pre_g)  # [B, h_l]
        f = tf.sigmoid(pre_f)  # [B, h_l]
        o = tf.sigmoid(pre_o)  # [B, h_l]

        z=tf.squeeze(z, axis=[1])
        z_b = tf.squeeze(z_b, axis=[1])

        c_next=tf.where(tf.equal(z,tf.constant(1,dtype=tf.float32)),
                 tf.multiply(i,g),#flush
                 tf.where(tf.equal(z_b,tf.constant(1,dtype=tf.float32)),
                          tf.add(tf.multiply(c,f),tf.multiply(i,g)),#update
                          tf.identity(c) #copy
                 )
                 )

        h_next=tf.where(tf.equal(z,tf.constant(1,dtype=tf.float32)),
                 tf.multiply(o, tf.tanh(c_next)),#flush
                 tf.where(tf.equal(z_b,tf.constant(1,dtype=tf.float32)),
                          tf.multiply(o, tf.tanh(c_next)),#update
                          tf.identity(h) #copy
                 )
                 )

        slope_multiplier = 1
        pre_z_next = tf.sigmoid(pre_z_next * slope_multiplier)
        g = tf.get_default_graph()
        with g.gradient_override_map({"Round": "Identity"}):
            z_next = tf.round(pre_z_next)



        out_state=HMLSTMStateTuple(c=c_next,h=h_next,z=z_next)

        h_next=tf.nn.dropout(h_next,keep_prob=self.keep_p)
        output=tf.concat([h_next,z_next],axis=1)
        return output,out_state,concat


