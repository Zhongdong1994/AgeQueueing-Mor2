# -*- coding: utf-8 -*-
"""
Created on Wed May 16 06:50:38 2018
Develop a queue engine
Consider a queue with two different packet size
Implement different scheduling policies
@author: Liang Huang
"""
import random
from math import pow

import numpy as np
from collections import deque
import time


class QUEUE(object):
    """docstring for QUEUE
    Queueing with priority users. The smaller value, the higher priority.
    Poisson arrival process: total arrival rate = arrival_rate; probability distribtion for different users = user_prob
    Deterministic service process: service rates for different users = mu
    """

    def __init__(self, Nuser=10000, arrival_rate=0.6, mu=1, mode='FCFS'):
        '''
        'Age_Inef_Tag': False for effective age decreasing; True for non-effective age decreasing
        'Block_tag': True if the packet is blocked
        'mode': 'FCFS', 'LCFS','FCFSPriority','LCFSPriority', 'FCFSSRPT', 'LCFSSRPT','FCFSSEA','LCFSSEA'
        'preemptive': 0 for non-preemptive, and 1 for preemptive
        '''
        super(QUEUE, self).__init__()
        self.Nuser = Nuser
        self.arrival_rate = arrival_rate
        # self.user_prob = user_prob
        self.num_user_type = 1
        self.mu = mu
        self.mode = mode
        self.preemptive = self.mode in ['PLCFS', 'PSJF', 'SRPT','PABS']
        self.i_depart = np.zeros(self.num_user_type, dtype=int)
        self.largest_inqueue_time=0  ### largest inqueue time among all departed jobs
        # self.i_depart_effective = np.zeros(self.num_user_type, dtype=int)
        self.last_depart = -1  # by default no customer departs
        self.i_serving = -1  # by default no customer under serving
        # array to store all queueing related performance metric
        self.Customer = np.zeros(self.Nuser, dtype=np.dtype([('Inqueue_Time', float),
                                                             ('Arrival_Intv', float),  # arrival interval time
                                                             ('Waiting_Intv', float),
                                                             ('Serve_Intv', float),
                                                             ('Work_Load', float),
                                                             ('Remain_Work_Load', float),
                                                             ('Dequeue_Intv', float),
                                                             ('Dequeue_Time', float),
                                                             ('Block_Tag', bool),
                                                             #     ('Block_Depth',int),
                                                             ('Queue_Number', int),
                                                             ('Response_Time', float),
                                                             ('Age_Arvl', float),
                                                             ('Age_Dept', float),
                                                             ('Age_Peak', float)]))
        #     ('Age_Inef_Tag',bool),
        #     ('Priority',int)]

        self.generate_arvl()
        # init queue for different priorities
        self.queues = []
        # suspended queue for preempted packets
        self.suspended_queues = []
        ### the queue with all effetive departures
        self.effe_queues = []
        self.conqueue=[]

    def reset(self):
        self.preemptive = self.mode in ['PLCFS', 'PSJF', 'SRPT','PABS']
        self.i_depart = np.zeros(self.num_user_type, dtype=int)
        # self.i_depart_effective = np.zeros(self.num_user_type, dtype=int)
        self.last_depart = -1  # by default no customer departs
        self.i_serving = -1  # by default no customer under serving
        self.largest_inqueue_time = 0
        Customer = np.zeros(self.Nuser, dtype=np.dtype([('Inqueue_Time', float),
                                                        ('Arrival_Intv', float),
                                                        ('Waiting_Intv', float),
                                                        ('Serve_Intv', float),
                                                        ('Work_Load', float),
                                                        ('Remain_Work_Load', float),  ### index=5
                                                        ('Dequeue_Intv', float),
                                                        ('Dequeue_Time', float),
                                                        ('Block_Tag', bool),
                                                        #    ('Block_Depth',int),
                                                        ('Queue_Number', int),
                                                        ('Response_Time', float),
                                                        ('Age_Arvl', float),
                                                        ('Age_Dept', float),
                                                        ('Age_Peak', float)]))
        #   ('Age_Inef_Tag',bool),
        #   ('Priority',int)
        Customer['Arrival_Intv'] = np.copy(self.Customer['Arrival_Intv'])
        # Customer['Priority'] = np.copy(self.Customer['Priority'])
        Customer['Work_Load'] = np.copy(self.Customer['Work_Load'])
        self.Customer = np.copy(Customer)
        self.Customer['Remain_Work_Load'] = np.copy(self.Customer['Work_Load'])
        # init queue for different priorities
        self.queues = []
        # suspended queue for preempted packets, only for self.preemptive=True
        self.suspended_queues = []
        self.effe_queues = []
        self.conqueue = []

    def generate_arvl(self):
        '''
        return arrival intervals with arrival_rate and index each customer's priority
        '''
        self.Customer['Arrival_Intv'] = np.random.exponential(1 / self.arrival_rate, size=self.Nuser)
        # self.Customer['Priority'] = np.random.choice(self.num_user_type, size=self.Nuser, p=self.user_prob)
        # print(self.Customer['Priority'])


        self.Customer['Work_Load'] = np.random.exponential(1 / self.mu, size=self.Nuser)
        #self.Customer['Work_Load'] = 0.29752*np.random.weibull(0.39837, size=self.Nuser)
        #self.Customer['Work_Load'] =  np.random.uniform(0,2, size=self.Nuser)


        # print(self.Customer['Work_Load'])
        # print((1/np.array(self.mu)))
        self.Customer['Remain_Work_Load'] = np.copy(self.Customer['Work_Load'])

    def arrive(self, i):
        ''' enqueue the i-th customer
        '''
        if i is 0:
            # enqueue the first customer; other parameters are as default values 0
            self.Customer['Inqueue_Time'][i] = self.Customer['Arrival_Intv'][i]
            self.largest_inqueue_time=self.Customer['Inqueue_Time'][i]
            self.Customer['Age_Arvl'][i] = self.Customer['Inqueue_Time'][i]
            # for future finite queue
            # self.Customer['Block_Depth'][i] = 1
        else:
            self.Customer['Inqueue_Time'][i] = self.Customer['Inqueue_Time'][i - 1] + self.Customer['Arrival_Intv'][i]
            # compute queue length upon the arrival of i-th customer
            self.Customer['Queue_Number'][i] = self.queue_len()
            # age upon the i-th arrival
            self.Customer['Age_Arvl'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Inqueue_Time'][
                i] - self.Customer['Dequeue_Time'][self.last_depart]

    def enqueue(self, i):
        # enqueue if the i-th customer is not blocked
        if self.Customer['Block_Tag'][i] == False:
            # enqueue customer with respect to its priority
            self.queue_append(i)

    def suspended_queue_len(self):
        ''' return suspended queue length
        '''
        return len(self.suspended_queues)

    def queue_len(self):
        ''' return current queue length
        '''
        return len(self.queues) + self.suspended_queue_len()

    def s_os(self, temp=1):  ### return the index of minimum original work load in the suspended queue
        minimum = 0
        for x in range(temp):
            if self.Customer['Work_Load'][self.suspended_queues[x]] < self.Customer['Work_Load'][
                self.suspended_queues[minimum]]:
                minimum = x
        return minimum

    def s_rs(self,
             temp=1):  ### return the index of minimum remaining work load in the suspended queue
        minimum = 0
        for x in range(temp):
            if self.Customer['Remain_Work_Load'][self.suspended_queues[x]] < self.Customer['Remain_Work_Load'][
                self.suspended_queues[minimum]]:
                minimum = x
        return minimum

    def os(self, temp=1):  ### return the index of minimum original work load in the queue
        minimum = 0
        for x in range(temp):
            if self.Customer['Work_Load'][self.queues[x]] < self.Customer['Work_Load'][self.queues[minimum]]:
                minimum = x
        return minimum

    def rs(self, temp=1):  ### return the index of minimum remaining work load in the queue
        minimum = 0
        for x in range(temp):
            if self.Customer['Remain_Work_Load'][self.queues[x]] < self.Customer['Remain_Work_Load'][
                self.queues[minimum]]:
                minimum = x
        return minimum

    def ageDropLargest(self,temp=1,currentTime=0):
        minimum = 0
        for x in range(temp):
            if currentTime+self.Customer['Remain_Work_Load'][self.queues[x]]-self.Customer['Inqueue_Time'][self.queues[x]] < \
                    currentTime+self.Customer['Remain_Work_Load'][self.queues[minimum]]-self.Customer['Inqueue_Time'][self.queues[minimum]]:
                minimum = x
        return minimum



    def queue_pop(self,currentTime=0):
        ''' pop one customer for service
        modes = ['FCFS', 'RANDOM','LCFS','PLCFS','SJF','PSJF','SRPT','ABS','PABS']
        '''
        # check preempted customer
        if self.preemptive and self.suspended_queue_len() > 0:
            if self.mode == 'PLCFS':
                temp = self.suspended_queues.pop()
                # print(temp)
                return temp
            if self.mode == 'PSJF':
                # print("working")
                if len(self.queues) == 0:
                    return self.suspended_queues.pop(self.s_os(len(self.suspended_queues)))
                elif self.Customer['Work_Load'][self.suspended_queues[self.s_os(len(self.suspended_queues))]] < \
                        self.Customer['Work_Load'][self.queues[self.os(len(self.queues))]]:
                    return self.suspended_queues.pop(self.s_os(len(self.suspended_queues)))
                else:
                    return self.queues.pop(self.os(len(self.queues)))

            if self.mode == 'SRPT':
                if len(self.queues) == 0:
                    return self.suspended_queues.pop(self.s_rs(len(self.suspended_queues)))
                elif self.Customer['Remain_Work_Load'][self.suspended_queues[self.s_rs(len(self.suspended_queues))]] < \
                        self.Customer['Remain_Work_Load'][self.queues[self.rs(len(self.queues))]]:
                    return self.suspended_queues.pop(self.s_rs(len(self.suspended_queues)))
                else:
                    return self.queues.pop(self.rs(len(self.queues)))
                # return self.suspended_queues.pop(self.s_rs(len(self.suspended_queues)))
        if self.queue_len() > 0:
            if self.mode == 'FCFS':
                return self.queues.pop(0)
            if self.mode == 'LCFS':
                return self.queues.pop()
            if self.mode == 'RANDOM':
                temp = random.randint(0, len(self.queues) - 1)
                # print(temp)
                return self.queues.pop(temp)
            if self.mode == 'SJF':
                return self.queues.pop(self.os(len(self.queues)))
            if self.mode == 'PLCFS':
                # print(self.queues)
                return self.queues.pop()
            if self.mode == 'PSJF':
                # print("working")
                return self.queues.pop(self.os(len(self.queues)))
            if self.mode == 'SRPT':
                return self.queues.pop(self.rs(len(self.queues)))
            if self.mode == 'PS':
                return self.queues.pop(self.rs(len(self.queues)))
            if self.mode=='ABS' or self.mode=='PABS':
                return self.queues.pop(self.ageDropLargest(len(self.queues), currentTime))
        return -1

    def queue_append(self, i):
        '''
        append one customer. Left ones goes out first, and right ones goes last
        modes = ['FCFS', 'RANDOM','LCFS','PS','PLCFS','FB','SJF','PSJF','SRPT','ABS]
        '''
        if self.mode in ['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT','ABS','PABS']:
            self.queues.append(i)
        else:
            print('Improper queueing mode in queue_append!', self.mode)

    def suspended_queue_append(self, i):
        '''
        append one preempted customer
        '''
        if self.mode in ['FCFS', 'RANDOM', 'LCFS', 'PS', 'PLCFS', 'SJF', 'PSJF', 'SRPT','ABS','PABS']:
            self.suspended_queues.append(i)
        else:
            print('Improper queueing mode in suspended_queue_append!', self.mode)

    def minAservice(self,temp=[]): # return the  one job in temp[] whose attained service time is least
            minimum = temp[0]
            for x in temp:
                if self.Customer['Serve_Intv'][x] < self.Customer['Serve_Intv'][minimum]:
                    minimum = x

            return minimum

    def minRservice(self,temp=[]): # return the one job in temp[] whose remaining service time is least
            minimum = temp[0]
            for x in temp:
                if self.Customer['Remain_Work_Load'][x] < self.Customer['Remain_Work_Load'][minimum]:
                    minimum = x
            return minimum

    def serve(self, i, t_begin, t_end):
        ''' serve the i-th customer
        return the time when the service ends/stops
        '''
        # idx_queue_minSer=self.minAservice(self.queues)
        # idx_conqueue_minRem=self.minRservice(self.conqueue)
        # if self.mode=='FB':
        #     T2=len(self.conqueue)*(self.Customer['Serve_Intv'][idx_queue_minSer]-self.Customer['Serve_Intv'][0])
        #     T1=len(self.conqueue)*(self.Customer['Remain_Work_Load'][idx_conqueue_minRem])
        #     if T2==0: # implies queue[]==conqueue[]


        if self.mode == 'PS':
            if t_end == -1 or self.Customer['Remain_Work_Load'][i] < (t_end - t_begin) / (len(self.queues) + 1):
                self.Customer['Serve_Intv'][i] += self.Customer['Remain_Work_Load'][i]
                self.Customer['Dequeue_Time'][i] = t_begin + self.Customer['Remain_Work_Load'][i] * (
                            len(self.queues) + 1)
                for j in self.queues:
                    self.Customer['Serve_Intv'][j] += self.Customer['Remain_Work_Load'][i]
                    self.Customer['Remain_Work_Load'][j] -= self.Customer['Remain_Work_Load'][i]
                self.Customer['Remain_Work_Load'][i] = 0
                return self.depart(i)
            else:
                # self.Customer['Serve_Intv'][i] += t_end - t_begin
                # self.Customer['Remain_Work_Load'][i] -= t_end - t_begin
                self.Customer['Serve_Intv'][i] += (t_end - t_begin)/ (len(self.queues) + 1)
                self.Customer['Remain_Work_Load'][i] -= (t_end - t_begin) / (len(self.queues) + 1)
                for j in self.queues:
                    self.Customer['Serve_Intv'][j] += (t_end - t_begin)/ (len(self.queues) + 1)
                    self.Customer['Remain_Work_Load'][j] -= (t_end - t_begin) / (len(self.queues) + 1)
                return t_end

        ### the following is for general case (i.e., the jobs are served one by one)
        if t_end == -1 or self.Customer['Remain_Work_Load'][i] < t_end - t_begin:
            # customer departs
            self.Customer['Serve_Intv'][i] += self.Customer['Remain_Work_Load'][i]
            # depart time = current time + work load
            self.Customer['Dequeue_Time'][i] = t_begin + self.Customer['Remain_Work_Load'][i]
            self.Customer['Remain_Work_Load'][i] = 0
            return self.depart(i)
        else:
            # part of work is served
            self.Customer['Serve_Intv'][i] += t_end - t_begin
            self.Customer['Remain_Work_Load'][i] -= t_end - t_begin

            return t_end

    def depart(self, i):
        '''
        the i-th customer departs
        update waiting time, depart interval, peak age, and age after depart
        '''
        # waiting time = depart time - arrival time - service time
        self.Customer['Response_Time'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i]
        self.Customer['Waiting_Intv'][i] = self.Customer['Response_Time'][i] - self.Customer['Serve_Intv'][i]
        self.Customer['Dequeue_Intv'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Dequeue_Time'][
            self.last_depart]

        # if self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i] > self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]:
        #     # ineffective departure
        #     # self.Customer['Age_Inef_Tag'][i] = True
        #     self.Customer['Age_Dept'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][i]
        if self.Customer['Inqueue_Time'][i] < self.largest_inqueue_time:
            self.Customer['Age_Dept'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][
                i]
        else:
            # effective departure
            self.largest_inqueue_time=self.Customer['Inqueue_Time'][i]
            self.Customer['Age_Dept'][i] = self.Customer['Dequeue_Time'][i] - self.Customer['Inqueue_Time'][i]
            self.Customer['Age_Peak'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][
                i]
            self.effe_queues.append(i)

        self.last_depart = i
        self.i_serving = -1

        return self.Customer['Dequeue_Time'][i]

    def serve_between_time(self, t_begin, t_end):
        t = t_begin
        # serve the current customer
        if self.i_serving >= 0:
            t = self.serve(self.i_serving, t, t_end)
        # when there is additional time to serve other customers
        while (t < t_end or t_end == -1) and self.queue_len() > 0:
            # next customer
            self.i_serving = self.queue_pop(t)  # here we get the index of next customer
            # serve the customer
            t = self.serve(self.i_serving, t, t_end)

    def is_preempted(self, i_old, i_new):
        '''
        return True is preemption
        modes = ['PLCFS','PSJF','SRPT']
        '''
        if self.mode == 'PLCFS':
            # always preempt
            return True
        elif self.mode == 'SRPT':
            # depends on the remaining workload
            # print("working")
            return self.Customer['Remain_Work_Load'][i_new] < self.Customer['Remain_Work_Load'][i_old]
        elif self.mode == 'PSJF':
            # compare the expected age, current time is the arrival time of i_new
            # the expected age of i_new is its work load
            temp = self.Customer['Work_Load'][i_new] < self.Customer['Work_Load'][i_old]
            # print(temp)
            return temp
        elif self.mode=='PABS':
             check= self.Customer['Remain_Work_Load'][i_new] - self.Customer['Inqueue_Time'][i_new]< \
                    self.Customer['Remain_Work_Load'][i_old] - self.Customer['Inqueue_Time'][i_old]
             return check
        else:
            return False


        # # no preemption by default
        # return False

    def preempt(self, i_old, i_new):
        '''
        i_old is preempted by i_new
        '''
        if self.mode=='PABS':
            self.queue_append(i_old)
            self.i_serving=i_new
        else:
            # suspend i_old
            self.suspended_queue_append(i_old)
            # set the new customer as serving
            self.i_serving = i_new


    def queueing(self):
        self.arrive(0)
        self.enqueue(0)
        # arrival index
        idx_a = 0
        # depart index
        idx_d = -1

        while idx_a < self.Nuser - 1:
            idx_a += 1

            # if self.mode=='FB':  working on...
            #     self.conqueue=[]
            #     self.conqueue.append(idx_a-1)

            self.serve_between_time(self.Customer['Inqueue_Time'][idx_a - 1],
                                    self.Customer['Inqueue_Time'][idx_a - 1] + self.Customer['Arrival_Intv'][idx_a])
            self.arrive(idx_a)

            if self.preemptive and self.is_preempted(self.i_serving, idx_a):
                self.preempt(self.i_serving, idx_a)
            elif self.mode == 'PS' and self.Customer['Remain_Work_Load'][self.i_serving] > \
                    self.Customer['Remain_Work_Load'][idx_a]:
                self.queues.append(self.i_serving)
                self.i_serving = idx_a
            # elif self.mode=='FB':  # this part is useful, do not delete
            #     if self.i_serving>=0:
            #         self.queues.append(self.i_serving)
            #         self.i_serving = idx_a
            #     else:
            #         self.i_serving = idx_a
            else:
                # no preemption, enqueue the customer
                self.queue_append(idx_a)

        # serve remaining customers in the queue till the end
        self.serve_between_time(self.Customer['Inqueue_Time'][idx_a], -1)

    def change_mode(self, mode):
        '''
        change mode and reset the queue. but keep the generate_arvl()
        '''
        self.mode = mode
        self.reset()

    # calculate those average performance metrics, we only use the last half customers after the queueing is stable
    def mean_age(self):
        '''
        the average age can be calculated from arriving age due to PASTA
        return: mean age
        '''
        total_age = 0
        for x in range(len(self.effe_queues)):
            if x == 0:
                total_age += self.Customer['Age_Peak'][self.effe_queues[x]] * self.Customer['Dequeue_Time'][
                    self.effe_queues[x]] - 0.5 * pow(self.Customer['Dequeue_Time'][self.effe_queues[x]], 2)
            else:
                temp = self.Customer['Dequeue_Time'][self.effe_queues[x]] - self.Customer['Dequeue_Time'][
                    self.effe_queues[x - 1]]
                total_age += self.Customer['Age_Peak'][self.effe_queues[x]] * temp - 0.5 * pow(temp, 2)
        return total_age / self.Customer['Dequeue_Time'][self.Nuser - 1]

    def mean_peak_age(self):
        '''
        the average peak age
        return: mean peak_age
        '''
        total_page = 0
        for x in self.effe_queues:
            total_page += self.Customer['Age_Peak'][x]
        return total_page / len(self.effe_queues)

    def mean_response_time(self):
        total_response_time = 0
        for x in range(len(self.Customer)):
            total_response_time += self.Customer['Response_Time'][x]
        return total_response_time / len(self.Customer)

    def mean_queue_len(self):
        '''
        the average queue length observed based on customer arrivals due to PASTA
        return: mean queue length
        '''
        return sum(self.Customer['Queue_Number'][int(self.Nuser / 2):] / (self.Nuser - int(self.Nuser / 2)))

    @property
    def parameters(self):
        return 'Nuser=' + str(self.Nuser) + ', arrival_rate=' + str(self.arrival_rate) + ', mu =' + str(
            self.mu) + ', mode =' + self.mode