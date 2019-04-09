#References: https://github.com/bolducp/hierarchical-rnn/tree/master/hmlstm
import HMSLSTM_Main as Main
import tensorflow as tf
import numpy as np
# from matplotlib import pyplot as plt

print(tf.__version__)





def regression_per_video(labels, predictions, starts_list,
                         idx):  # starts_list includes the index of the start index in each video in the main input file
    #VRE
    labels_pool = np.array([0, 5, 10])
    np.clip(predictions, 0, 10, out=predictions)
    LOSS = 0
    for i, start in enumerate(starts_list):
        if (i + 1) == len(starts_list):
            predicts = predictions[start:]
            Y = labels[start:]
        else:
            predicts = predictions[start:starts_list[i + 1]]
            Y = labels[start:starts_list[i + 1]]

        ave_predicts = np.mean(predicts)
        predicted_index = (ave_predicts // 3.34).astype(np.int8)
        final_decision = labels_pool[predicted_index]
        if Y[0, 0] == 0:
            L = 3.3
        if Y[0, 0] == 10:
            L = 6.6

        if final_decision == Y[0, 0]:
            loss = 0

        else:
            if Y[0,0]==5 and ave_predicts<5:
                loss=(ave_predicts - 3.3) ** 2
            elif Y[0, 0] == 5 and ave_predicts >= 5:
                loss = (ave_predicts - 6.6) ** 2
            else:
                loss = (ave_predicts - L) ** 2

        LOSS = loss + LOSS

    if idx % 15 == 0 or idx==79:
        print('Per Video Regression Error is :%f ' % (LOSS / len(starts_list)))

    return LOSS / len(starts_list)  # returnes the accuracy per video


def regression_per_blink(labels, predictions, starts_list,
                         idx):  # starts_list includes the index of the start index in each video in the main input file
    #BSRE
    labels_pool = np.array([0, 5, 10])
    np.clip(predictions, 0, 10, out=predictions)
    LOSS = 0
    for i, start in enumerate(starts_list):
        if (i + 1) == len(starts_list):
            predicts = predictions[start:]
            Y = labels[start:]
        else:
            predicts = predictions[start:starts_list[i + 1]]
            Y = labels[start:starts_list[i + 1]]
        if Y[0, 0]==0:
            L=3.3
        if Y[0, 0] == 10:
            L = 6.6

        predicted_index = (predicts // 3.34).astype(np.int8)
        predicted_labels = labels_pool[predicted_index]
        if Y[0,0]==5:
            loss=np.sum((predicts[np.logical_and(predicted_labels != Y[0, 0],predicted_labels<5)] - 3.3) ** 2)
            loss=loss+np.sum((predicts[np.logical_and(predicted_labels != Y[0, 0],predicted_labels >= 5)] - 6.6) ** 2)
        else:
            loss = np.sum((predicts[predicted_labels != Y[0, 0]] - L) ** 2)
        LOSS = LOSS + loss

    if idx % 15 == 0 or idx==79:
        print('Per Blink Sequence Regression Error is :%f ' % (LOSS / len(labels)))

    return LOSS / len(labels)


def vote_accuracy_per_video(labels,predictions,starts_list,idx): #starts_list includes the index of the start index in each video in the main input file
    #VA
    count=0
    labels_pool = np.array([0, 5, 10])
    np.clip(predictions, 0, 10, out=predictions)
    for i,start in enumerate(starts_list):
        if (i+1)==len(starts_list):
            predicts = predictions[start:]
            Y = labels[start:]
        else:
            predicts=predictions[start:starts_list[i+1]]
            Y=labels[start:starts_list[i+1]]
        predicted_index = (predicts // 3.34).astype(np.int8)
        predicted_labels = labels_pool[predicted_index]
        alert_voted_percent=np.sum(predicted_labels==labels_pool[0])/len(Y)
        lowVigilant_voted_percent = np.sum(predicted_labels == labels_pool[1]) / len(Y)
        drowsy_voted_percent = np.sum(predicted_labels == labels_pool[2]) / len(Y)
        max=np.max([alert_voted_percent, lowVigilant_voted_percent, drowsy_voted_percent])

        if np.sum([alert_voted_percent, lowVigilant_voted_percent, drowsy_voted_percent]==max)!=1:
            ave_predicts=np.mean(predicts)
            predicted_index = (ave_predicts // 3.34).astype(np.int8)
            final_decision = labels_pool[predicted_index]
        else:
            final_decision=labels_pool[np.argmax([alert_voted_percent,lowVigilant_voted_percent,drowsy_voted_percent])]
        if final_decision==Y[0,0]:
            count=count+1
        if idx%10==0:
            print(str(i+1)+': '+'True label is :%d and the detected label is %d' %(Y[0,0],final_decision))

    return count/len(starts_list)  #returnes the accuracy per video





def calc_accuracy_per_batch(Y, predicts):  #Y_size=[Batch_size,1]
    labels_pool = np.array([0, 5, 10])
    np.clip(predicts,0,10,out=predicts)
    predicted_index = (predicts // 3.34).astype(np.int8)
    predicted_labels = labels_pool[predicted_index]
    is_correct = np.equal(predicted_labels, Y.astype(np.int8))
    accuracy = np.sum(is_correct)/len(is_correct)

    return accuracy

def batchNorm(x,beta,gamma,training,scope='bn'):
    with tf.variable_scope(scope):
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def Network(input,Pre_fc1_size,Post_fc1_size_per_layer,embb_size,embb_size2,Post_fc2_size,hstate_size,num_layers,feature_size,
            step_size,output_size,keep_p,training):
    #input :[Batch,step_size,features]
    #hstate_size=list of hstate_szie for each layer  [layers]

    end_points = {}
    batch_size = tf.shape(input)[0]
    with tf.variable_scope('pre_fc1'):
        pre_fc1_weights=tf.get_variable('weights',[feature_size,Pre_fc1_size],dtype=tf.float32,
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False,seed=None,dtype=tf.float32))
        pre_fc1_biases = tf.get_variable('biases', [Pre_fc1_size], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))


        reshaped_input_net=tf.reshape(input, [-1, feature_size])
        input_RNN=tf.matmul(reshaped_input_net,pre_fc1_weights)
        input_RNN = batchNorm(input_RNN, pre_fc1_biases, None, training, scope='bn')
        input_RNN=tf.nn.relu(input_RNN)
        input_RNN=tf.reshape(input_RNN,[-1,step_size,Pre_fc1_size]) # size=[batch,Time,Pre_fc1_size ]
        input_RNN=tf.nn.dropout(input_RNN,keep_p)




    end_points['pre_fc1']=input_RNN


    hmslstm_block=Main.HMSLSTM_Block(input_size=[batch_size,step_size,Pre_fc1_size],step_size=step_size,
                                     hstate_size=hstate_size,num_layers=num_layers,keep_p=keep_p)

    output_RNN_set,states_RNN,concati=hmslstm_block(input_RNN,reuse=False)
    end_points['mid_layers'] = output_RNN_set
    with tf.variable_scope('post_fc1'):

        for lay in range(num_layers):
            post_fc1_weights = tf.get_variable('weights_%s' % lay, [hstate_size[lay], Post_fc1_size_per_layer], dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                                dtype=tf.float32))
            post_fc1_biases = tf.get_variable('biases_%s' % lay, [Post_fc1_size_per_layer], dtype=tf.float32,
                                              initializer=tf.constant_initializer(0.0))
            trash,output_RNN=tf.split(output_RNN_set[lay],[step_size-1,1],axis=0,name='layers')  #size of output_RNN[lay] is (step,batch,hsize),
            #  but we want just the last step
            output_RNN=tf.squeeze(output_RNN,axis=0) #size=(batch,h_size)
            post_fc1 = tf.matmul(output_RNN, post_fc1_weights)
            post_fc1 = batchNorm(post_fc1, post_fc1_biases,None, training, scope='bn')

            if lay==0:
                post_fc1_out=post_fc1
            else:
                post_fc1_out=tf.concat([post_fc1_out,post_fc1],axis=1) #size=[Batch,layer*Post_fc1_size_per_layer]

        post_fc1_out=tf.nn.relu(post_fc1_out)
        post_fc1_out = tf.nn.dropout(post_fc1_out,keep_p)
        end_points['post_fc1'] = post_fc1_out
    with tf.variable_scope('embeddings'):
        embeddings_weights = tf.get_variable('weights' , [Post_fc1_size_per_layer*num_layers,embb_size], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                              dtype=tf.float32))
        embeddings_biases = tf.get_variable('biases' , [embb_size], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))



        embeddings = tf.matmul(post_fc1_out, embeddings_weights)
        embeddings = batchNorm(embeddings, embeddings_biases, None, training, scope='bn')
        embeddings = tf.nn.relu(embeddings)
        embeddings = tf.nn.dropout(embeddings, keep_p)
        end_points['embeddings'] = embeddings
    with tf.variable_scope('embeddings2'):
        embeddings_weights2 = tf.get_variable('weights' , [embb_size,embb_size2], dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                              dtype=tf.float32))
        embeddings_biases2 = tf.get_variable('biases' , [embb_size2], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))



        embeddings2 = tf.matmul(embeddings, embeddings_weights2)
        embeddings2 = batchNorm(embeddings2, embeddings_biases2, None, training, scope='bn')
        embeddings2 = tf.nn.relu(embeddings2)
        embeddings2 = tf.nn.dropout(embeddings2, keep_p)
        end_points['embeddings2'] = embeddings2

    with tf.variable_scope('post_fc2'):
        post_fc2_weights = tf.get_variable('weights' , [embb_size2, Post_fc2_size],
                                                 dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                            dtype=tf.float32))
        post_fc2_biases = tf.get_variable('biases', [Post_fc2_size], dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.0))

        post_fc2_out = tf.matmul(embeddings2 , post_fc2_weights)
        post_fc2_out = batchNorm(post_fc2_out,post_fc2_biases, None, training, scope='bn')
        post_fc2_out=tf.nn.relu(post_fc2_out)
        post_fc2_out = tf.nn.dropout(post_fc2_out, keep_p)
        end_points['post_fc2'] = post_fc2_out
    with tf.variable_scope('last_fc'):
        last_fc_weights = tf.get_variable('weights' , [Post_fc2_size,output_size],
                                           dtype=tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                           dtype=tf.float32))
        last_fc_biases = tf.get_variable('biases', [output_size], dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))

        output = tf.matmul(post_fc2_out, last_fc_weights)+last_fc_biases


        if output_size==1:
            end_points['before the last sigmoid'] = output
            output = 10 * tf.sigmoid(output)


    return output,end_points,concati #size=[Batch,1]



########
########

def batch_gen(data,label,batch_size):  # data=[Total data points, T,F]   #label=[Total Data Points, 1]
    n=len(data)
    batch_num=n // batch_size
    for b in range(batch_num):  # Here it generates batches of data within 1 epoch consecutively
        X=data[batch_size*b:batch_size*(b+1),:,:]
        Y= label[batch_size * b:batch_size * (b + 1),:]
        yield X,Y
    if n> batch_size * (b + 1):
        X = data[batch_size * (b + 1):, :, :]
        Y = label[batch_size * (b + 1):, :]
        yield X, Y



def epoch_gen(data,label,batch_size,num_epochs): # data=[Total data points, T,F]  # This generates epochs of batches of data
    for epoch in range(num_epochs): # Inside one epoch
        yield batch_gen(data,label,batch_size)

def save_variables(sess,path,f):
        saver = tf.train.Saver()
        print('saving variables...\n')
        saver.save(sess,path+'my_model%d'%f)

def Train(total_input,total_labels,TestB,TestL,output_size,feature_size,batch_size,num_epochs,Pre_fc1_size,Post_fc1_size_per_layer,embb_size,
          embb_size2,Post_fc2_size,hstate_size,num_layers,step_size,drop_out_p,lr,th,start_i,load,fold_num):  #total_input is the shuffled input with size=[Total data points, T,F]

    #shape of total_input [N,T,F]
    #shape of total_labels=[N,1]
    #TestB:the test blink sequences
    #TestL: the test labels
    #feature_size:input feature_size
    #Order of layers==>  Pre_fc1---HMLSTM---Post_fc1----embb----embb2----Post_fc2---out
    #step_size:the time steps of the HMLSTM network
    #lr:leanining rate
    #th: the Delta for the cost function
    #start_i: the start indices of blink sequences in each test video
    #load: if True loads the weights from disk[Binary]
    #fold_num: decides that the model used is the one trained on all the folds except fold_num (if load==True)
            #  decides that the test model (if load==False)
    tf.reset_default_graph()
    L2loss=0
    input_net = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='bacth_in')
    labels = tf.placeholder(tf.float32, shape=(None, output_size), name='labels_net')  #size=[batch,1]
    keep_p=tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool,name='phase_train')
    output,end_points,concati=Network(input=input_net,Pre_fc1_size=Pre_fc1_size,Post_fc1_size_per_layer=Post_fc1_size_per_layer,
                   embb_size=embb_size,embb_size2=embb_size2,Post_fc2_size=Post_fc2_size,hstate_size=hstate_size,num_layers=num_layers,
                   feature_size=feature_size,step_size=step_size,output_size=output_size,keep_p=keep_p,training=training)
    error=tf.abs(output-labels)
    loss2 =tf.maximum(0.0,tf.square(error)-th)
    loss2 = tf.reduce_mean(loss2)
    variable_path='./'
    with tf.variable_scope('last_fc',reuse=True):
        last_fc_weights = tf.get_variable('weights')
    with tf.variable_scope('post_fc2',reuse=True):
        post_fc2_weights = tf.get_variable('weights')
    with tf.variable_scope('embeddings',reuse=True):
        embeddings_weights = tf.get_variable('weights')
    with tf.variable_scope('embeddings2',reuse=True):
        embeddings_weights2 = tf.get_variable('weights')
    with tf.variable_scope('pre_fc1',reuse=True):
        pre_fc1_weights = tf.get_variable('weights')

    with tf.variable_scope('post_fc1',reuse=True):
        for lay in range(num_layers):
            post_fc1_weights = tf.get_variable('weights_%s' % lay)
            L2loss=tf.nn.l2_loss(post_fc1_weights)+L2loss
    #
    loss=loss2+0.1 * (tf.nn.l2_loss(last_fc_weights) +tf.nn.l2_loss(pre_fc1_weights) + L2loss+
                       tf.nn.l2_loss(post_fc2_weights) + tf.nn.l2_loss(embeddings_weights)+ tf.nn.l2_loss(embeddings_weights2))
    optimizer=tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        ###
        if(load==True):
            saver = tf.train.Saver()
            print('loading variables...')
            saver.restore(sess, variable_path+'my_model%d'%fold_num)
        else:
            sess.run(tf.global_variables_initializer())
        ####plotting_setup\
        y = np.zeros([num_epochs])
        yy = np.zeros([num_epochs])
        y_test = np.zeros([num_epochs])
        yy_test = np.zeros([num_epochs])
        y_v = np.zeros([num_epochs])
        x=np.linspace(1, num_epochs, num_epochs, endpoint=True)
        ####
        for idx,epoch in enumerate(epoch_gen(data=total_input,label=total_labels,batch_size=batch_size,num_epochs=num_epochs)):#set of batches in one epoch
            loss_per_epoch = 0
            sum=0
            if load==False:
                for b,(X,Y) in enumerate(epoch):# each batch in each epoch

                        if output_size==1:
                            _,loss_values,predicts,mid_values,concat=sess.run([optimizer,loss,output,end_points,concati],
                                                                           feed_dict={input_net: X, labels: Y,keep_p:drop_out_p,training:True})
                            accuracy = calc_accuracy_per_batch(Y, predicts) #BSA

                        loss_per_epoch= loss_values +loss_per_epoch
                        sum=sum+accuracy                                # calculating the moving average for acc
                        moving_accuracy=sum/(b+1)
                        if b%50==0:
                            print('Epoch number: ' + str(idx) + ' ---- ' + 'Batch number ' + str(
                                b) + ' -- the loss is: ' + str(loss_values) + '--the accuracy: '+str(moving_accuracy)+'\n')
            ###########TEsting
            if output_size==1:
                loss_values_Test, predicts_Test,mid_vT = sess.run([loss, output,end_points],feed_dict={input_net: TestB, labels: TestL,keep_p:1.0,training:False})
                accuracy_Test = calc_accuracy_per_batch(TestL, predicts_Test) #BSA
                accuracy_per_videoV=vote_accuracy_per_video(TestL,predicts_Test,start_i,idx) #VA
            ###########
            if load==False:
                print("For Training: "+str(loss_per_epoch)+" , "+str(moving_accuracy))
                yy[idx] = moving_accuracy
                y[idx] = loss_per_epoch
            print("BSA: " + str(loss_values_Test) + " , " + str(accuracy_Test))

            print("VA: " + str(accuracy_per_videoV))

            regression_per_blink(TestL,predicts_Test,start_i,idx) #BSRE
            regression_per_video(TestL, predicts_Test, start_i, idx) #VRE
            print("----------------------------------------------")


            yy_test[idx]=accuracy_Test
            y_test[idx]=loss_values_Test
            y_v[idx]=accuracy_per_videoV
            if load==True:
                break
            if idx % 5 == 0:
                print("Sav")
        if load==False:
            save_variables(sess,variable_path,fold_num) # Saving every epoch



    return x,y,yy,y_test,yy_test,y_v





################################################
load=False  #Load the weights from disk if True
for i in range(5): #Cross validation but recommended to run each fold a few times to see the best perfomrance as you may
    # get caught up in a local minimum

    ii=i    # ii decides the model and i decides the fold_num for test
    if load==True:
        ii=i+1
    Blinks = np.load('Blinks_30_Fold%d.npy'%(i+1))
    Labels = np.load('Labels_30_Fold%d.npy'%(i+1))
    BlinksTest = np.load('BlinksTest_30_Fold%d.npy'%(i+1))
    LabelsTest = np.load('./LabelsTest_30_Fold%d.npy'%(i+1))
    #deciding the indices of each video based on the fold
    #####################Normalizing the input#############Second phase
    BlinksTest[:,:,0]=(BlinksTest[:,:,0]-np.mean(Blinks[:,:,0]))/np.std(Blinks[:,:,0])
    Blinks[:,:,0]=(Blinks[:,:,0]-np.mean(Blinks[:,:,0]))/np.std(Blinks[:,:,0])
    #####
    #####
    BlinksTest[:,:,1]=(BlinksTest[:,:,1]-np.mean(Blinks[:,:,1]))/np.std(Blinks[:,:,1])
    Blinks[:,:,1]=(Blinks[:,:,1]-np.mean(Blinks[:,:,1]))/np.std(Blinks[:,:,1])
    #####
    BlinksTest[:,:,2]=(BlinksTest[:,:,2]-np.mean(Blinks[:,:,2]))/np.std(Blinks[:,:,2])
    Blinks[:,:,2]=(Blinks[:,:,2]-np.mean(Blinks[:,:,2]))/np.std(Blinks[:,:,2])
    #####
    BlinksTest[:,:,3]=(BlinksTest[:,:,3]-np.mean(Blinks[:,:,3]))/np.std(Blinks[:,:,3])
    Blinks[:,:,3]=(Blinks[:,:,3]-np.mean(Blinks[:,:,3]))/np.std(Blinks[:,:,3])
    ####
    ####JUST TO DOUBLE CHECK
    ####
    # print(np.mean(Blinks[:,:,0]),np.mean(Blinks[:,:,1]),np.mean(Blinks[:,:,2]),np.mean(Blinks[:,:,3]))
    # print(np.std(Blinks[:,:,0]),np.std(Blinks[:,:,1]),np.std(Blinks[:,:,2]),np.std(Blinks[:,:,3]))
    # print(np.mean(BlinksTest[:,:,0]),np.mean(BlinksTest[:,:,1]),np.mean(BlinksTest[:,:,2]),np.mean(BlinksTest[:,:,3]))
    # print(np.std(BlinksTest[:,:,0]),np.std(BlinksTest[:,:,1]),np.std(BlinksTest[:,:,2]),np.std(BlinksTest[:,:,3]))

    # start indices are the indices in BlinksTest where the set of blink sequences in each video starts. It is used for VA, BSRE and BSA  calculation
    #These numbers are specifically for the "Blinks" and "BlinksTest" arrays in my case. If you generate the blinks from scratch,
    # The numbers should be re-adjusted.
    #NOTE: These indices are only used to compute three evaluation metrics of the test set.
    if i==0:
        start_indices=[0,52,106,156,164,182,219,224,303,318,330,398,443,471,521,615,657,693,718,753,879,948,949,965,
                       1068,1091,1126,1177,1178,1185,1200,1221,1226] # for step_size=30

        # start_indices=[0,37,76,111,112,115,137,138,202,203,204,257,287,300,335,414,441,462,472,492,603,657,658,659,
        #                747,755,775,811,812,813,814,820,821] # for step_size=60
        # start_indices=[0,59,121,178,193,218,262,274,360,383,402,477,530,565,622,723,772,815,847,889,1023,
        #                1100,1109,1132,1243,1273,1316,1374,1375,1389,1411,1440,1452]      # for step_size=15
        # start_indices = [0,7,16,21,22,23,24,25,59,60,61,84,85,86,91,140,141,142,143,144,225,249,250,251,309,310,
        #                  311,317,318,319,320,321,322] # for step size=120

    if i==1:
        start_indices=[0,1,104,126,155,168,268,339,491,695,696,697,712,715,761,818,886,973,1012,1087,1189,1270,1298,
                       1320,1380,1397,1406,1429,1439,1463,1464,1465,1549,1592,1753,1945]  # for step_size=30

        # start_indices=[0,1,89,96,110,111,196,252,389,578,579,580,581,582,613,655,708,780,804,864,951,1017,1030,1037,
        #                1082,1084,1085,1093,1094,1103,1104,1105,1174,1202,1348,1525] # for step_size=60

        # start_indices = [0,7,118,148,184,205,312,390,549,760,761,769,792,803,856,921,997,1091,1138,1220,1329,
        #                  1418,1454,1484,1552,1576,1593,1623,1640,1672,1679,1686,1777,1828,1996,2195]  # for step_size=15
    if i==2:
        start_indices=[0,1,2,7,11,19,36,71,89,97,263,505,740,779,888,995,1129,1232,1435,1441,1496,1502,1503,1524,
                       1617,1618,1662,1755,1790,1881,2039,2065,2138,2183,2184,2190] # for step_size=30

        # start_indices=[0,1,2,3,4,5,7,27,30,31,182,409,629,653,747,839,958,1046,1234,1235,1275,1276,1277,1283,1361,
        #     1362,1391,1469,1489,1565,1708,1719,1777]# for step_size=60
        # start_indices = [0,1,4,17,28,43,68,111,136,152,325,575,817,863,979,1093,1235,1345,1556,1569,1632,1646,1648,
        #                  1676,1777,1778,1830,1931,1973,2071,2236,2270,2351]  # for step_size=15

        # start_indices = [0,1,2,3,4,5,6,7,8,9,130,327,517,518,582, 644,733,791,949,950,960,961,962,963,
        #                  1011,1012,1013,1061,1062,1108,1221,1222,1250]  # for step size=120

    if i==3:
        start_indices=[0,8,38,108,113,114,135,137,285,380,411,473,568,569,581,590,610,699,819,827,865,884,885,923,949,990,
                       1051,1156,1157,1185,1265,1266,1284,1299,1391,1450]# for step_size=30

        # start_indices=[0,1,16,71,72,73,79,80,213,293,309,356,436,437,438,439,444,518,623,624,647,651,652,675,686,
        #                712,758,848,849,862,927,928,931,932,1009,1053] # for step_size=60
        # start_indices = [0,16,54,131,143,144,172,182,337,440,479,549,651,658,678,695,722,818,946,961,1007,
        #                  1033,1038,1083,1117,1165,1233,1345,1347,1383,1470,1472,1498,1520,1619,1685] # for step_size=15

        # start_indices=[0,1,2,27,28,29,30,31,134,184,185,202,252,253,254,255,256,300,375,376,377,378,379,380,381,382,398,
        #                458,459,460,495,496,497,498,545,559]# for step_size=120

    if i==4:
        start_indices=[0,1,27,45,55,153,205,256,382,515,521,522,527,536,556,575,598,730,794,795,904,955,999,
                       1027,1059,1060,1088,1143,1275,1480,1605,1622,1659]# for step_size=30

        # start_indices=[0,1,12,15,16,99,136,172,283,401,402,403,404,405,410,414,422,539,588,589,683,719,748,761,778,779,
        #                792,832,949,1139,1249,1251,1273] # for step_size=60
        # start_indices = [0,1,35,61,78,183,243,301,434,575,588,591,604,621,648,675,706,845,917,922,1039,1097,
        #                  1149,1185,1224,1227,1262,1325,1465,1677,1810,1834,1878]# for step_size=15

        # start_indices = [0,1,2,3,4,57,64,70,151,239,240,241,242,243,244,245,246,333,352,353,417,423,424,425,426,427,
        #                  428,438,525,685,765,766,767]# for step_size=120
    print('######################')
    print(i)
    print('######################')
    start_indices=np.asarray(start_indices)
    x,loss,accuracy,loss_Test,accuracy_Test,acc_per_Vid=Train(total_input=Blinks,total_labels=Labels,TestB=BlinksTest,TestL=LabelsTest,
                    output_size=1,feature_size=4,batch_size=64,num_epochs=80,Pre_fc1_size=32,Post_fc1_size_per_layer=16,
                    embb_size=16,embb_size2=16,Post_fc2_size=8,hstate_size=[32,32,32,32],num_layers=4,step_size=30,drop_out_p=1.0,
                                                  lr=0.000053,th=1.253,start_i=start_indices,load=load,fold_num=ii)


    if load==False:
        np.save(open('./x%d.npy' %ii, 'wb'),x)
        np.save(open('./loss%d.npy'%ii, 'wb'),loss) #for training
        np.save(open('./accuracy%d.npy' %ii, 'wb'),accuracy) #for training
        np.save(open('./loss%dTest.npy'%ii, 'wb'),loss_Test) #for test
        np.save(open('./accuracy%dTest.npy'%ii, 'wb'),accuracy_Test) #for test (BSA)
        np.save(open('./accuracy%dVTest.npy'%ii, 'wb'),acc_per_Vid) #for test    (VA)


