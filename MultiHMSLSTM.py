#reference: https://github.com/n-s-f/hierarchical-rnn/tree/master/hmlstm
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf


class MultiHMSLSTM(rnn_cell_impl.RNNCell):
    def __init__(self,cells,reuse):
        super().__init__(_reuse=reuse)
        self.cells=cells


    @property
    def output_size(self):
        return self.cells[-1].output_size

    @property
    def state_size(self):
        return tuple(c.state_size for c in self.cells)

    def zero_state(self):
        return tuple([cell.zero_state() for cell in self.cells])

    def call(self,raw_inputs,states):  #raw_inputs=[input,z]
        out_state=[]
        out_set=[]
        states=states[1] # we just want the full_states
        h_aboves=[s.h for s in states[1:len(states)]]
        h_aboves.append(tf.zeros([self.cells[0].batch_size,self.cells[len(states)-1].h_above_size]))
        # the uppermost cell but it does not matter because we wont use it, we only need it for compatibility reasons
        for i,cell in enumerate (self.cells):
            with vs.variable_scope("cell_%d" % i):
                inputs=tf.concat([h_aboves[i],raw_inputs],axis=1)  #h_above,h_below,z
                out,state,concati=cell(inputs,states[i])
                if i==0:
                    concat=concati
                out_set.append(out[:,0:cell.hstate_size])
                raw_inputs=out
                out_state.append(state)





        # outputs=tuple([s.h for s in out_state])
        outputs=tuple(out_set)
        out_state=tuple(out_state)
        return outputs,out_state,concati
