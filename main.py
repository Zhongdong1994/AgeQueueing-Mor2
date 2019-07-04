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
    # arrival_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    arrival_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    #arrival_rates = [0.95]
    Nuser =100000
    mu = 1
    rounds = 1
    # modes = ['FCFS', 'RANDOM','LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT','ABS','PADS','MPSJF', 'MSRPT','MPADS']  ###  FB is currently unavailable
    # modes = ['PSJF', 'SRPT',  'PADS', 'MPSJF', 'MSRPT', 'MPADS','PSJFE','SRPTE', 'MPADS2','PADF','MPADF','MPADF2','ADM','PADM']
    # modes = ['FCFS', 'RANDOM','LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT','ADS','PADS','SRPTL','ADF','PADF']
    # modes=['ADS','ADF','ADM','PADS','PADF','PADM','MPADS','MPADF','MPADM2']
    # modes=['FCFS', 'RANDOM','LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF','SRPT','SRPT3','MSRPT','SRPTE']
    # modes = ['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'MPSJF', 'PSJFE', 'SRPT', 'SRPTA', 'MSRPT',
    #          'SRPTE', 'ADS', 'PADS', 'MPADS', 'MPADS2', 'ADF',
    #          'PADF', 'MPADF', 'MPADF2', 'ADM', 'PADM', 'MPADM', 'MPADM2']
    modes=['AoI2']

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
    for i in range(len(arrival_rates)):
        print(arrival_rates[i])
        Mean = compare(Nuser, arrival_rates[i], mu, modes)
        if i == 0:
            Mean2 = Mean  ### Mean2 is to store all data (the data generated in all loops)
        else:
            tempMean = pd.DataFrame(data)
            tempMean['age'] = arrival_rates[i]
            Mean2 = Mean2.append(tempMean)
            Mean2 = Mean2.append(Mean)

        for j in range(rounds - 1):
            temp = compare(Nuser, arrival_rates[i], mu, modes)
            tempMean = pd.DataFrame(data)
            tempMean['age'] = arrival_rates[i]
            tempMean['peak'] = j + 1
            Mean2 = Mean2.append(tempMean)
            Mean2 = Mean2.append(temp)
            Mean += temp

        Mean = Mean.multiply(1 / rounds)
        if i == 0:
            Mean1 = Mean  ### Mean1 is to store all averagered data (is exactly same as the output)
        else:
            tempMean = pd.DataFrame(data)
            tempMean['age'] = arrival_rates[i]
            Mean1 = Mean1.append(tempMean)
            Mean1 = Mean1.append(Mean)
        print(Mean)
        # store simulation data in results.h5
        with pd.HDFStore('results.h5') as store:
            store.put(str(arrival_rates[i]), Mean)

    Mean1.to_csv("Averagedata.csv")
    Mean2.to_csv("Originaldata.csv")

    # plot curves
    import matplotlib.pyplot as plt

    # mean response
    fig, ax = plt.subplots(figsize=(15, 8))
    with pd.HDFStore('results.h5') as store:
        for m in range(len(modes)):
            plt.plot(arrival_rates,
                     [store[str(arrival_rates[i])]['response'][modes[m]] for i in range(len(arrival_rates))])

        plt.ylabel('average response')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        ax.set_ylim(
            [0, 9])
        plt.show()

    # mean age
    fig, ax = plt.subplots(figsize=(15, 8))
    with pd.HDFStore('results.h5') as store:
        # plt.plot(arrival_rates, [store[str(arrival_rates[i])]['age'][modes[5]] for i in range(len(arrival_rates))])
        for m in range(len(modes)):
            plt.plot(arrival_rates, [store[str(arrival_rates[i])]['age'][modes[m]] for i in range(len(arrival_rates))])

        plt.ylabel('average age')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        #ax.set_ylim([0, max([store[str(arrival_rates[i])]['age'][modes[1]] for i in range(len(arrival_rates))]) * 1.1])
        plt.show()
    # peak age
    fig, ax = plt.subplots(figsize=(15, 8))
    with pd.HDFStore('results.h5') as store:
        # plt.plot(arrival_rates, [store[str(arrival_rates[i])]['peak'][modes[5]] for i in range(len(arrival_rates))])
        for m in range(len(modes)):
            plt.plot(arrival_rates, [store[str(arrival_rates[i])]['peak'][modes[m]] for i in range(len(arrival_rates))])

        plt.ylabel('average peak age')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        #ax.set_ylim([0, max([store[str(arrival_rates[i])]['peak'][modes[1]] for i in range(len(arrival_rates))]) * 1.1])
        plt.show()


def compare(Nuser=1000, arrival_rate=0.35, mu=1,
            modes=['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT', 'ADS', 'PADS']):
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

    queue = QUEUE(Nuser, arrival_rate, mu)

    # print(queue.parameters)
    # print(arrival_rate)
    for i in range(len(modes)):
        queue.change_mode(modes[i])
        queue.queueing()
        Mean['age'][queue.mode] = queue.mean_age()
        Mean['peak'][queue.mode] = queue.mean_peak_age()
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



