# -*- coding: utf-8 -*-
"""
Created on Wed May 16 06:50:38 2018
Implement different simulations/tests
@author: Liang Huang
"""

import numpy as np
import pandas as pd
import time

# plt.switch_backend('agg')
from queueengine import QUEUE


def simulate():
    '''
    run simulation for different arrival rates and plot curves
    '''

    #arrival_rates = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.87,0.9]   #zipf, mu=4
    #arrival_rates = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.72]       #zipf, mu=3
    arrival_rates = [0.1] #zipf
    Nuser =pow(10,5)  # fixed
    N = [13,22,34,48,65,84,106,130,157]; # Number of  Elements
    expn = [2.0116,2.1836,2.2665,2.3144,2.3452,2.3666,2.3823,2.3942,2.4035]; # Exponent
    Csquare = [1,1.5,2,2.5,3,3.5,4,4.5,5]
    mu = 0.5
    rounds = 1


    #modes = ['FCFS', 'RANDOM','LCFS', 'TSLS','PLCFS', 'SJF', 'PSJF', 'SRPT','ADM','ADS','ADF','PADM','PADS','PADF']
    #modes = ['RANDOM','LCFS','SJF','SJF2']
    modes = ['RANDOM','LCFS','SJF']
    #modes=['SRPT','LCFS','SJF','ADM','ADS','ADF','AoI2', 'AoI3']

    #modes = [ 'FCFS', 'RANDOM','LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT','RANDOME', 'LCFSE', 'SJFE', 'PSJFE', 'SRPTE','SRPTA']
    #modes=['SRPT','SRPTE', 'ADF','PADF','ADS','PADS','ADM','PADM']

    '''
    SRPTL policy is very similar to SRPT policy expect that SRPTL chooses the job that has latest arrival time when multiple jobs have the
    same remaining time

    MPSJF, MSRPT and MPADS policies are modified version of PSJF, SRPT and PADS. The difference between these modified polices and original 
    policies is: once the preemption happens, those modified policies will drop all outdated jobs. Motivations: Outdated jobs sometimes will block new
    jobs and the AoI/PAoI benefits nothing from serving the outdated jobs.

    PSJFE, SRPTE and PADSE will kick out the outdated jobs both when the preemption happens and when normal departure happens. 
    '''

    data = np.zeros(1, dtype=np.dtype([('age', float),
                                       ('peak', float),
                                       ('len', float),
                                       ('response', float)]))
    for i in range(len(Csquare)):
        print(Csquare[i])
        Mean = compare(Nuser, arrival_rates[0], mu, modes,N[i],expn[i],1)
        if i == 0:
            Mean2 = Mean  ### Mean2 is to store all data (the data generated in all loops)
        else:
            tempMean = pd.DataFrame(data)
            tempMean['age'] = arrival_rates[0]
            Mean2 = Mean2.append(tempMean)
            Mean2 = Mean2.append(Mean)

        for j in range(rounds - 1):
            print("rounds:", j+1)
            temp = compare(Nuser, arrival_rates[0], mu, modes,N[i],expn[i],j+2)
            tempMean = pd.DataFrame(data)
            tempMean['age'] = arrival_rates[0]
            tempMean['peak'] = j + 1
            Mean2 = Mean2.append(tempMean)
            Mean2 = Mean2.append(temp)
            Mean += temp

        Mean = Mean.multiply(1 / rounds)
        if i == 0:
            Mean1 = Mean  ### Mean1 is to store all averagered data (is exactly same as the output)
        else:
            tempMean = pd.DataFrame(data)
            tempMean['age'] = arrival_rates[0]
            Mean1 = Mean1.append(tempMean)
            Mean1 = Mean1.append(Mean)
        print(Mean)
        # store simulation data in results.h5
        with pd.HDFStore('results.h5') as store:
            store.put(str(Csquare[i]), Mean)

    Mean1.to_csv("Averagedata.csv")
    Mean2.to_csv("Originaldata.csv")

    # plot curves
    import matplotlib.pyplot as plt

    # mean response
    # fig, ax = plt.subplots(figsize=(15, 8))
    # with pd.HDFStore('results.h5') as store:
    #     for m in range(len(modes)):
    #         plt.plot(arrival_rates,
    #                  [store[str(arrival_rates[i])]['response'][modes[m]] for i in range(len(arrival_rates))])
    #     plt.ylabel('average response')
    #     plt.xlabel('arrival rates')
    #     plt.legend(modes)
    #     ax.set_ylim(
    #         [0, 9])
    #     plt.show()
    #
    # # mean age
    # fig, ax = plt.subplots(figsize=(15, 8))
    # with pd.HDFStore('results.h5') as store:
    #     # plt.plot(arrival_rates, [store[str(arrival_rates[i])]['age'][modes[5]] for i in range(len(arrival_rates))])
    #     for m in range(len(modes)):
    #         plt.plot(arrival_rates, [store[str(arrival_rates[i])]['age'][modes[m]] for i in range(len(arrival_rates))])
    #
    #     plt.ylabel('average age')
    #     plt.xlabel('arrival rates')
    #     plt.legend(modes)
    #     #ax.set_ylim([0, max([store[str(arrival_rates[i])]['age'][modes[1]] for i in range(len(arrival_rates))]) * 1.1])
    #     plt.show()
    # # peak age
    # fig, ax = plt.subplots(figsize=(15, 8))
    # with pd.HDFStore('results.h5') as store:
    #     # plt.plot(arrival_rates, [store[str(arrival_rates[i])]['peak'][modes[5]] for i in range(len(arrival_rates))])
    #     for m in range(len(modes)):
    #         plt.plot(arrival_rates, [store[str(arrival_rates[i])]['peak'][modes[m]] for i in range(len(arrival_rates))])
    #
    #     plt.ylabel('average peak age')
    #     plt.xlabel('arrival rates')
    #     plt.legend(modes)
    #     #ax.set_ylim([0, max([store[str(arrival_rates[i])]['peak'][modes[1]] for i in range(len(arrival_rates))]) * 1.1])
    #     plt.show()


def compare(Nuser=1000, arrival_rate=0.35, mu=1,
            modes=['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT', 'ADS', 'PADS'],popIndx=10, expIndex=2,roundIndx=1):
    '''
    compare different scheduling modes
    '''
    # modes = ['FCFS','RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT','ADS','PADS']
    data = np.zeros(len(modes), dtype=np.dtype([('age', float),
                                                ('peak', float),
                                                ('len', float),
                                                ('response', float)]))
    # print(data)
    Mean = pd.DataFrame(data, index=modes)
    # print(Mean)

    queue = QUEUE(Nuser, arrival_rate, mu,'FCFS',popIndx,expIndex,roundIndx)

    # print(queue.parameters)
    # print(arrival_rate)
    for i in range(len(modes)):
        queue.change_mode(modes[i])
        print("complete mode "+modes[i])
        queue.queueing()
        Mean['age'][queue.mode] = queue.mean_age()
        Mean['peak'][queue.mode] = queue.mean_peak_age(popIndx, expIndex,roundIndx)
        Mean['len'][queue.mode] = queue.mean_queue_len()
        Mean['response'][queue.mode] = queue.mean_response_time()
        # Mean['ineff_dept'][queue.mode] = sum(queue.Customer['Age_Inef_Tag'] == True)/queue.Nuser

    # print(Mean)
    return Mean


def test():
    queue = QUEUE(Nuser=1000,
                  arrival_rate=0.3,
                  user_prob=[0.5, 0.5],
                  mu=[0.8, 0.2],
                  mode='FCFSSRPT')
    queue.queueing()
    print(queue.parameters)
    print(queue.Customer.dtype.names)
    print(queue.Customer)
    print("Current scheduling mode:", queue.mode)
    print("Mean response time", queue.mean_response_time())
    print("Mean age:", queue.mean_age())
    print("Mean peak age:", queue.mean_peak_age())
    print("Mean queue length:", queue.mean_queue_len())

    # number of ineffective departure
    print("% Ineffective departure:", sum(queue.Customer['Age_Inef_Tag'] == True) / queue.Nuser)


if __name__ == '__main__':
    start_time = time.time()
    simulate()
    #    compare()
    #    test()
    total_time = time.time() - start_time
    print('time_cost:%s' % total_time)



