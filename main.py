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
    arrival_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    Nuser = 20000
    #   user_prob=[0.5, 0.5]
    mu = 1
    rounds =50
    modes = ['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT']  ###  FB is currently unavailable

    # for i in range(len(arrival_rates)):
    #     Mean = compare(Nuser, arrival_rates[i], mu, modes)
    #     # store simulation data in results.h5
    #     with pd.HDFStore('results.h5') as store:
    #         store.put(str(arrival_rates[i]), Mean)

    for i in range(len(arrival_rates)):
        print(arrival_rates[i])
        Mean = compare(Nuser, arrival_rates[i], mu, modes)
        for j in range(rounds - 1):
            Mean += compare(Nuser, arrival_rates[i], mu, modes)
            # store simulation data in results.h5
        Mean = Mean.multiply(1 / rounds)
        print(Mean)
        with pd.HDFStore('results.h5') as store:
            store.put(str(arrival_rates[i]), Mean)

    # plot curves
    import matplotlib.pyplot as plt

    # mean response
    fig, ax = plt.subplots(figsize=(15, 8))
    with pd.HDFStore('results.h5') as store:
        for m in range(len(modes)):
            plt.plot(arrival_rates,
                     [store[str(arrival_rates[i])]['response'][modes[m]] for i in range(len(arrival_rates))])

        plt.ylabel('mean response')
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

        plt.ylabel('mean age')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        ax.set_ylim([0, max([store[str(arrival_rates[i])]['age'][modes[1]] for i in range(len(arrival_rates))]) * 1.1])
        plt.show()
    # peak age
    fig, ax = plt.subplots(figsize=(15, 8))
    with pd.HDFStore('results.h5') as store:
        # plt.plot(arrival_rates, [store[str(arrival_rates[i])]['peak'][modes[5]] for i in range(len(arrival_rates))])
        for m in range(len(modes)):
            plt.plot(arrival_rates, [store[str(arrival_rates[i])]['peak'][modes[m]] for i in range(len(arrival_rates))])

        plt.ylabel('peak age')
        plt.xlabel('arrival rates')
        plt.legend(modes)
        ax.set_ylim([0, max([store[str(arrival_rates[i])]['peak'][modes[1]] for i in range(len(arrival_rates))]) * 1.1])
        plt.show()

    # # queue length
    # fig, ax = plt.subplots(figsize=(15,8))
    # with pd.HDFStore('results.h5') as store:
    #     for m in range(len(modes)):
    #         plt.plot(arrival_rates,[store[str(arrival_rates[i])]['len'][modes[m]] for i in range(len(arrival_rates))] )
    #
    #     plt.ylabel('queue length')
    #     plt.xlabel('arrival rates')
    #     plt.legend(modes)
    #     plt.show()

    # ineffective depart ratio
    # fig, ax = plt.subplots(figsize=(15,8))
    # with pd.HDFStore('results.h5') as store:
    #     for m in range(len(modes)):
    #         plt.plot(arrival_rates,[store[str(arrival_rates[i])]['ineff_dept'][modes[m]] for i in range(len(arrival_rates))] )
    #
    #     plt.ylabel('Ineffective departure ratio')
    #     plt.xlabel('arrival rates')
    #     plt.legend(modes)
    #     plt.show()


def compare(Nuser=1000, arrival_rate=0.35, mu=1,
            modes=['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT']):
    '''
    compare different scheduling modes
    '''
    modes = ['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT']
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
        # print(modes[i])
        # print(queue.Customer.dtype.names)
        # print(queue.Customer)
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




