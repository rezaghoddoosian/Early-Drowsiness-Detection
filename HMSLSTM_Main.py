#reference: https://github.com/n-s-f/hierarchical-rnn/tree/master/hmlstm
import HMSLSTM_cell as H
import MultiHMSLSTM as M
import tensorflow as tf

class HMSLSTM_Block():
    def __init__(self,input_size,step_size,hstate_size,num_layers,keep_p):
        self.input_size=input_size   #[Batch, steps,features]
        self.step_size=step_size
        self.hstate_size=hstate_size
        self.num_layers=num_layers
        self.batch_size=self.input_size[0]
        self.keep_p=keep_p

    def __call__(self, input,reuse): #input=[Batch, steps,features]
        cells=[]

        for layer in range(self.num_layers):
            if layer == 0:
                cell = H.HMSLSTM_cell(self.hstate_size[layer],self.input_size[2],
                                    self.hstate_size[layer + 1], self.batch_size,self.keep_p,reuse)

                cells.append(cell)

            elif layer == self.num_layers - 1:
                cell = H.HMSLSTM_cell(self.hstate_size[layer],  self.hstate_size[layer - 1],
                                    self.hstate_size[0], self.batch_size,self.keep_p,reuse)

                cells.append(cell)
            else:
                cell = H.HMSLSTM_cell(self.hstate_size[layer],self.hstate_size[layer - 1],
                                    self.hstate_size[layer + 1], self.batch_size,self.keep_p,reuse)

                cells.append(cell)

        MultiCell = M.MultiHMSLSTM(cells,reuse)

        def build_the_Multi_block(s,i): # i =[Batch, features]

            z=tf.ones([self.batch_size,1])
            inp=tf.concat([i,z],axis=1)
            return  MultiCell(inp, s)

        initial_s=MultiCell.zero_state()

        initial_o=[s.h for s in initial_s]
        initial_o=tuple(initial_o)
        concati_initial=tf.ones([self.batch_size,4*tf.shape(initial_o[0])[1]+1])
        initial=(initial_o,initial_s,concati_initial)
        inp2=tf.transpose(input, [1, 0, 2])
        output,states,concat=tf.scan(build_the_Multi_block,inp2,initializer=initial)




        return output,states,concat



