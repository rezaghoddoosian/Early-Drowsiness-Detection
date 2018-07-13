import os
import numpy as np

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def normalize_blinks(num_blinks, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur, Vel, u_Vel,
                     sigma_Vel):
    # input is the blinking features as well as their mean and std, output is a [num_blinksx4] matrix as the normalized blinks
    normalized_blinks = np.zeros([num_blinks, 4])
    normalized_Freq = (Freq[0:num_blinks] - u_Freq) / sigma_Freq
    normalized_blinks[:, 0] = normalized_Freq
    normalized_Amp = (Amp[0:num_blinks]  - u_Amp) / sigma_Amp
    normalized_blinks[:, 1] = normalized_Amp
    normalized_Dur = (Dur[0:num_blinks]  - u_Dur) / sigma_Dur
    normalized_blinks[:, 2] = normalized_Dur
    normalized_Vel = (Vel[0:num_blinks]  - u_Vel) / sigma_Vel
    normalized_blinks[:, 3] = normalized_Vel

    return normalized_blinks


def unroll_in_time(in_data, window_size, stride):
    # in_data is [n,4]            out_data is [N,Window_size,4]
    n = len(in_data)
    if n <= window_size:
        out_data = np.zeros([1, window_size, 4])
        out_data[0, -n:, :] = in_data
        return out_data
    else:
        N = ((n - window_size) // stride) + 1
        out_data = np.zeros([N, window_size, 4])
        for i in range(N):
            if i * stride + window_size <= n:
                out_data[i, :, :] = in_data[i * stride:i * stride + window_size, :]
            else:  # this line should not ever be executed because of the for mula used above N is the exact time the loop is executed
                break

        return out_data

def gen(folder_list,window_size,stride,path1):
    for ID, folder in enumerate(folder_list):
        print("#########\n")
        print(str(ID) + '-' + str(folder) + '\n')
        print("#########\n")
        files_per_person = os.listdir(path1 + '/' + folder)
        for txt_file in files_per_person:
            if txt_file == 'alert.txt':
                alertTXT = path1 + '/' + folder + '/' + txt_file
                Freq = np.loadtxt(alertTXT, usecols=1)
                Amp = np.loadtxt(alertTXT, usecols=2)
                Dur = np.loadtxt(alertTXT, usecols=3)
                Vel = np.loadtxt(alertTXT, usecols=4)
                blink_num = len(Freq)
                bunch_size = blink_num // 3  # one third used for baselining
                remained_size = blink_num - bunch_size
                # Using the last bunch_size number of blinks to calculate mean and std
                u_Freq = np.mean(Freq[-bunch_size:])
                sigma_Freq = np.std(Freq[-bunch_size:])
                if sigma_Freq == 0:
                    sigma_Freq = np.std(Freq)
                u_Amp = np.mean(Amp[-bunch_size:])
                sigma_Amp = np.std(Amp[-bunch_size:])
                if sigma_Amp == 0:
                    sigma_Amp = np.std(Amp)
                u_Dur = np.mean(Dur[-bunch_size:])

                sigma_Dur = np.std(Dur[-bunch_size:])
                if sigma_Dur == 0:
                    sigma_Dur = np.std(Dur)
                u_Vel = np.mean(Vel[-bunch_size:])
                sigma_Vel = np.std(Vel[-bunch_size:])
                if sigma_Vel == 0:
                    sigma_Vel = np.std(Vel)
                print('freq: %f, amp: %f, dur: %f, vel: %f \n' % (u_Freq, u_Amp, u_Dur, u_Vel))
                normalized_blinks = normalize_blinks(remained_size, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp,
                                                     Dur, u_Dur, sigma_Dur,
                                                              Vel, u_Vel, sigma_Vel)

                print('Postfreq: %f, Postamp: %f, Postdur: %f, Postvel: %f \n' % (np.mean(normalized_blinks[:, 0]),
                                                                                  np.mean(normalized_blinks[:, 1]),
                                                                                  np.mean(normalized_blinks[:, 2]),
                                                                                  np.mean(normalized_blinks[:, 3])))

                alert_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                # sweep a window over the blinks to chunk
                alert_labels = 0 * np.ones([len(alert_blink_unrolled), 1])

            if txt_file == 'semisleepy.txt':
                blinksTXT = path1 + '/' + folder + '/' + txt_file
                Freq = np.loadtxt(blinksTXT, usecols=1)
                Amp = np.loadtxt(blinksTXT, usecols=2)
                Dur = np.loadtxt(blinksTXT, usecols=3)
                Vel = np.loadtxt(blinksTXT, usecols=4)
                blink_num = len(Freq)

                normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
                                                     u_Dur, sigma_Dur, Vel, u_Vel, sigma_Vel)
                print('SEMIfreq: %f, SEMIamp: %f, SEMIdur: %f, SEMIvel: %f \n' % (np.mean(normalized_blinks[:, 0]),
                                                                                  np.mean(normalized_blinks[:, 1]),
                                                                                  np.mean(normalized_blinks[:, 2]),
                                                                                  np.mean(normalized_blinks[:, 3])))

                semi_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                semi_labels = 5 * np.ones([len(semi_blink_unrolled), 1])

            if txt_file == 'sleepy.txt':
                blinksTXT = path1 + '/' + folder + '/' + txt_file
                Freq = np.loadtxt(blinksTXT, usecols=1)
                Amp = np.loadtxt(blinksTXT, usecols=2)
                Dur = np.loadtxt(blinksTXT, usecols=3)
                Vel = np.loadtxt(blinksTXT, usecols=4)
                blink_num = len(Freq)

                normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
                                                     u_Dur, sigma_Dur, Vel, u_Vel, sigma_Vel)
                print(
                'SLEEPYfreq: %f, SLEEPYamp: %f, SLEEPYdur: %f, SLEEPYvel: %f \n' % (np.mean(normalized_blinks[:, 0]),
                                                                                    np.mean(normalized_blinks[:, 1]),
                                                                                    np.mean(normalized_blinks[:, 2]),
                                                                                    np.mean(normalized_blinks[:, 3])))

                sleepy_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                sleepy_labels = 10 * np.ones([len(sleepy_blink_unrolled), 1])

        tempX = np.concatenate((alert_blink_unrolled, semi_blink_unrolled, sleepy_blink_unrolled), axis=0)
        tempY = np.concatenate((alert_labels, semi_labels, sleepy_labels), axis=0)
        if ID > 0:
            output = np.concatenate((output, tempX), axis=0)
            labels = np.concatenate((labels, tempY), axis=0)
        else:
            output = tempX
            labels = tempY
    return output,labels



def Preprocess(path1,window_size,stride,test_fold):
    #path1 is the address to the folder of all subjects, each subject has three txt files for alert, semisleepy and sleepy levels
    #window_size decides the length of blink sequence
    #stride is the step by which the moving windo slides over consecutive blinks to generate the sequences
    #test_fold is the number of fold that is picked as test and uses the other folds as training
    #output=[N,T,F]
    path=path1
    folds_list = os.listdir(path1)
    for f, fold in enumerate(folds_list):
        print(fold)
        path1 = path + '/' + fold
        folder_list = os.listdir(path1)
        if fold==test_fold:
            outTest,labelTest=gen(folder_list,window_size,stride,path1)
            print("Not this fold ;)")
            continue
        for ID,folder in enumerate(folder_list):
            print("#########\n")
            print(str(ID)+'-'+ str(folder)+'\n')
            print("#########\n")
            files_per_person = os.listdir(path1 + '/' + folder)
            for txt_file in files_per_person:
                if txt_file=='alert.txt':
                    alertTXT = path1 + '/' + folder + '/' + txt_file
                    Freq = np.loadtxt(alertTXT, usecols=1)
                    Amp = np.loadtxt(alertTXT, usecols=2)
                    Dur = np.loadtxt(alertTXT, usecols=3)
                    Vel = np.loadtxt(alertTXT, usecols=4)
                    blink_num=len(Freq)
                    bunch_size=blink_num // 3   #one third used for baselining
                    remained_size=blink_num-bunch_size
                    # Using the last bunch_size number of blinks to calculate mean and std
                    u_Freq=np.mean(Freq[-bunch_size:])
                    sigma_Freq=np.std(Freq[-bunch_size:])
                    if sigma_Freq==0:
                        sigma_Freq=np.std(Freq)
                    u_Amp=np.mean(Amp[-bunch_size:])
                    sigma_Amp=np.std(Amp[-bunch_size:])
                    if sigma_Amp==0:
                        sigma_Amp=np.std(Amp)
                    u_Dur=np.mean(Dur[-bunch_size:])

                    sigma_Dur=np.std(Dur[-bunch_size:])
                    if sigma_Dur==0:
                        sigma_Dur=np.std(Dur)
                    u_Vel=np.mean(Vel[-bunch_size:])
                    sigma_Vel=np.std(Vel[-bunch_size:])
                    if sigma_Vel==0:
                        sigma_Vel=np.std(Vel)
                    print('freq: %f, amp: %f, dur: %f, vel: %f \n' %(u_Freq,u_Amp,u_Dur,u_Vel))
                    normalized_blinks=normalize_blinks(remained_size, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur,
                                     Vel, u_Vel, sigma_Vel)

                    print('Postfreq: %f, Postamp: %f, Postdur: %f, Postvel: %f \n' % (np.mean(normalized_blinks[:,0]),
                                                                                      np.mean(normalized_blinks[:,1]),
                                                                                      np.mean(normalized_blinks[:,2]),
                                                                                      np.mean(normalized_blinks[:,3])))

                    alert_blink_unrolled=unroll_in_time(normalized_blinks,window_size,stride)
                    # sweep a window over the blinks to chunk
                    alert_labels = 0 * np.ones([len(alert_blink_unrolled), 1])




                if txt_file=='semisleepy.txt':
                    blinksTXT = path1 + '/' + folder + '/' + txt_file
                    Freq = np.loadtxt(blinksTXT, usecols=1)
                    Amp = np.loadtxt(blinksTXT, usecols=2)
                    Dur = np.loadtxt(blinksTXT, usecols=3)
                    Vel = np.loadtxt(blinksTXT, usecols=4)
                    blink_num = len(Freq)


                    normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
                                                     u_Dur, sigma_Dur,Vel, u_Vel, sigma_Vel)
                    print('SEMIfreq: %f, SEMIamp: %f, SEMIdur: %f, SEMIvel: %f \n' % (np.mean(normalized_blinks[:,0]),
                                                                                      np.mean(normalized_blinks[:,1]),
                                                                                      np.mean(normalized_blinks[:,2]),
                                                                                      np.mean(normalized_blinks[:,3])))

                    semi_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                    semi_labels = 5* np.ones([len(semi_blink_unrolled), 1])

                if txt_file == 'sleepy.txt':
                    blinksTXT = path1 + '/' + folder + '/' + txt_file
                    Freq = np.loadtxt(blinksTXT, usecols=1)
                    Amp = np.loadtxt(blinksTXT, usecols=2)
                    Dur = np.loadtxt(blinksTXT, usecols=3)
                    Vel = np.loadtxt(blinksTXT, usecols=4)
                    blink_num = len(Freq)

                    normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
                                                         u_Dur, sigma_Dur, Vel, u_Vel, sigma_Vel)
                    print('SLEEPYfreq: %f, SLEEPYamp: %f, SLEEPYdur: %f, SLEEPYvel: %f \n'  % (np.mean(normalized_blinks[:,0]),
                                                                                      np.mean(normalized_blinks[:,1]),
                                                                                       np.mean(normalized_blinks[:,2]),
                                                                                      np.mean(normalized_blinks[:,3])))

                    sleepy_blink_unrolled = unroll_in_time(normalized_blinks, window_size, stride)
                    sleepy_labels=10*np.ones([len(sleepy_blink_unrolled),1])


            tempX=np.concatenate((alert_blink_unrolled,semi_blink_unrolled,sleepy_blink_unrolled),axis=0)
            tempY = np.concatenate((alert_labels, semi_labels, sleepy_labels), axis=0)
            if test_fold!="Fold1":
                start=0
            else:
                start=1
            if f !=start  or ID>0:
                output=np.concatenate((output,tempX),axis=0)
                labels=np.concatenate((labels,tempY),axis=0)
            else:
                output=tempX
                labels=tempY

    output,labels=unison_shuffled_copies(output,labels)
    print('We have %d training datapoints!!!' %len(labels))
    print('We have %d test datapoints!!!' % len(labelTest))
    print('We have in TOTAL %d datapoints!!!' % (len(labelTest)+len(labels)))
    return output,labels,outTest,labelTest

#path1 is the address to the folder of all subjects, each subject has three txt files for alert, semisleepy and sleepy levels
path1='Drowsiness Data'
window_size=30
stride=2
Training='./Blinks_30_Fold3.npy'
Testing='./BlinksTest_30_Fold3.npy'
#################Normalizing with respect to different individuals####First Phase
blinks,labels,blinksTest,labelTest=Preprocess(path1,window_size,stride,test_fold='Fold3')
# np.save(open(Training,'wb'),blinks)
# np.save(open('./Labels_30_Fold3.npy', 'wb'),labels)
np.save(open(Testing, 'wb'),blinksTest)
np.save(open('./LabelsTest_30_Fold3.npy', 'wb'),labelTest)

