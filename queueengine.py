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

    def __init__(self, Nuser=10000, arrival_rate=0.6, mu=1, mode='FCFS',popIndx=10, expIndex=2,roundIndx=1):
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
        self.preemptive = self.mode in ['PLCFS', 'PSJF', 'SRPT','PADS','MPSJF', 'MSRPT','MPADS','PSJFE','SRPTE','MPADS2','SRPTL','SRPTA',
                                        'PADF','MPADF','MPADF2','PADM','MPADM','MPADM2','AoI2PE','AoI3PE','AoI2RP','AoI3RP','PADFE','PADSE']
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
                                                             ('TSLS', float), # time-since-last-service
                                                             ('Block_Tag', bool),
                                                             #     ('Block_Depth',int),
                                                             ('Queue_Number', int),
                                                             ('Response_Time', float),
                                                             ('Age_Arvl', float),
                                                             ('Age_Dept', float),
                                                             ('Age_Peak', float)]))

        self.generate_arvl(popIndx,expIndex,roundIndx)
        # init queue for different priorities
        self.queues = []
        # suspended queue for preempted packets
        #self.suspended_queues = []
        ### the queue with all effetive departures
        self.effe_queues = []
        self.conqueue=[]
        self.effe_departure=[]

    def reset(self):
        self.preemptive = self.mode in ['PLCFS', 'PSJF', 'SRPT','PADS','MPSJF', 'MSRPT','MPADS','PSJFE','SRPTE','MPADS2','SRPTL','SRPTA',
                                        'PADF','MPADF','MPADF2','PADM','MPADM','MPADM2','AoI2PE','AoI3PE','AoI2RP','AoI3RP','PADFE','PADSE']
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
                                                        ('TSLS', float),  # time-since-last-service
                                                        ('Block_Tag', bool),
                                                        #    ('Block_Depth',int),
                                                        ('Queue_Number', int),
                                                        ('Response_Time', float),
                                                        ('Age_Arvl', float),
                                                        ('Age_Dept', float),
                                                        ('Age_Peak', float)]))
        Customer['Arrival_Intv'] = np.copy(self.Customer['Arrival_Intv'])
        # Customer['Priority'] = np.copy(self.Customer['Priority'])
        Customer['Work_Load'] = np.copy(self.Customer['Work_Load'])
        self.Customer = np.copy(Customer)
        self.Customer['Remain_Work_Load'] = np.copy(self.Customer['Work_Load'])
        # init queue for different priorities
        self.queues = []
        # suspended queue for preempted packets, only for self.preemptive=True
        #self.suspended_queues = []
        self.effe_queues = []
        self.conqueue = []
        self.effe_departure = []

    def generate_arvl(self,popIndx=10, expIndex=2,roundIndx=1):
        '''
        return arrival intervals with arrival_rate and index each customer's priority
        '''


        self.Customer['Arrival_Intv'] = np.random.geometric(self.arrival_rate, size=self.Nuser)
        #self.Customer['Arrival_Intv']=[3, 2, 5, 1, 2, 1, 1, 1, 1, 1]
        #print(self.Customer['Arrival_Intv'])
        # self.Customer['Priority'] = np.random.choice(self.num_user_type, size=self.Nuser, p=self.user_prob)
        # print(self.Customer['Priority'])



        #self.Customer['Work_Load']= np.random.geometric(self.mu, size=self.Nuser)
        loadpath='data/N='+str(popIndx)+'_s='+str(expIndex)+'_round='+str(roundIndx)+'.txt'
        loadtemp=np.loadtxt(loadpath)
        self.Customer['Work_Load'] = loadtemp
        #self.Customer['Work_Load']=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #print(self.Customer['Work_Load'])
        #self.Customer['Work_Load'] = 0.29752*np.random.weibull(0.39837, size=self.Nuser)  #  mean=1, csqr=10
        #self.Customer['Work_Load'] = 0.324414* np.random.weibull(0.41134, size=self.Nuser)  # mean=1, csqr=9
        #self.Customer['Work_Load'] = 0.356264 * np.random.weibull(0.426775, size=self.Nuser)  # mean=1, csqr=8
        #self.Customer['Work_Load'] = 0.394549 * np.random.weibull(0.445573, size=self.Nuser)  # mean=1, csqr=7
        #self.Customer['Work_Load'] = 0.441402 * np.random.weibull(0.469165, size=self.Nuser)  # mean=1, csqr=6
        #self.Customer['Work_Load'] = 0.5 * np.random.weibull(0.5, size=self.Nuser)  # mean=1, csqr=5
        #self.Customer['Work_Load'] = 0.57525 * np.random.weibull(0.542693, size=self.Nuser)  # mean=1, csqr=4
        #self.Customer['Work_Load'] = 0.674996 * np.random.weibull(0.607248, size=self.Nuser)  # mean=1, csqr=3
        #self.Customer['Work_Load'] = 0.811794 * np.random.weibull(0.720905, size=self.Nuser)  # mean=1, csqr=2
        #self.Customer['Work_Load'] = 1 * np.random.weibull(1, size=self.Nuser)  # mean=1, csqr=1

        #self.Customer['Work_Load'] =  np.random.uniform(0,2, size=self.Nuser)
        #self.Customer['Work_Load']=0.9*np.ones(self.Nuser)

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
            self.Customer['Queue_Number'][i] = len(self.queues)
            # age upon the i-th arrival
            self.Customer['Age_Arvl'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Inqueue_Time'][
                i] - self.Customer['Dequeue_Time'][self.last_depart]

    def enqueue(self, i):
        # enqueue if the i-th customer is not blocked
        if self.Customer['Block_Tag'][i] == False:
            # enqueue customer with respect to its priority
            self.queue_append(i)




    def os(self, temp=1):  ### return the index of minimum original work load in the queue
        minimum = 0
        for x in range(temp):
            if self.Customer['Work_Load'][self.queues[x]] <= self.Customer['Work_Load'][self.queues[minimum]]:
                minimum = x
        return minimum

    def rs(self, temp=1):  ### return the index of a job  which has the  minimum remaining work load in the queue, FCFS if two jobs have the same minimum remaining work load
        minimum = 0
        for x in range(temp):
            if self.Customer['Remain_Work_Load'][self.queues[x]] <= self.Customer['Remain_Work_Load'][
                self.queues[minimum]]:
                minimum = x
        return minimum

    def tsls(self,temp=1):
        maximum = 0
        for x in range(temp):
            if self.Customer['TSLS'][self.queues[x]] >= self.Customer['TSLS'][
                self.queues[maximum]]:
                maximum = x
        return maximum

    def rs2(self, temp=1):  ### return the index of a job  which has the  minimum remaining work load in the queue, LCFS if two jobs have the same minimum remaining work load
        minimum = 0 #temp-1
        for x in range(temp):
            if self.Customer['Remain_Work_Load'][self.queues[x]] < self.Customer['Remain_Work_Load'][
                self.queues[minimum]]:
                minimum =x
            # if self.Customer['Remain_Work_Load'][self.queues[temp-1-x]] < self.Customer['Remain_Work_Load'][
            #     self.queues[minimum]]:
            #     minimum = temp-1-x
        return minimum

    def ageDroptoSmallest(self,temp=1,currentTime=0):
        jobsMakeAgeDrop = []
        jobsNotMakeAgeDrop = []
        for x in range(temp):
            if self.Customer['Inqueue_Time'][self.queues[x]] > self.largest_inqueue_time:
                jobsMakeAgeDrop.append(self.queues[x])
            else:
                jobsNotMakeAgeDrop.append(self.queues[x])
        if len(jobsMakeAgeDrop) == 0:
            return self.rs(len(self.queues))
        else:
            minimum = jobsMakeAgeDrop[0]
            for x in jobsMakeAgeDrop:
                if currentTime+self.Customer['Remain_Work_Load'][x]-self.Customer['Inqueue_Time'][x] <= \
                     currentTime+self.Customer['Remain_Work_Load'][minimum]-self.Customer['Inqueue_Time'][minimum]:
                    minimum = x
            return self.queues.index(minimum)

    def ageDropFast(self,temp=1):
        jobsMakeAgeDrop=[]
        jobsNotMakeAgeDrop = []
        for x in range(temp):
            if self.Customer['Inqueue_Time'][self.queues[x]] > self.largest_inqueue_time:
                jobsMakeAgeDrop.append(self.queues[x])
            else:
                jobsNotMakeAgeDrop.append(self.queues[x])
        if len(jobsMakeAgeDrop)==0:
            return self.rs(len(self.queues))
        else:
            minimum=jobsMakeAgeDrop[0]
            for x in jobsMakeAgeDrop:
                if self.Customer['Remain_Work_Load'][x] <= self.Customer['Remain_Work_Load'][minimum]:
                    minimum = x
            return self.queues.index(minimum)

    def ageDropMost(self,temp=1):
        jobsMakeAgeDrop=[]
        jobsNotMakeAgeDrop = []
        for x in range(temp):
            if self.Customer['Inqueue_Time'][self.queues[x]] > self.largest_inqueue_time:
                jobsMakeAgeDrop.append(self.queues[x])
            else:
                jobsNotMakeAgeDrop.append(self.queues[x])
        if len(jobsMakeAgeDrop)==0:
            return self.rs(len(self.queues))
        else:
            minimum=jobsMakeAgeDrop[0]
            for x in jobsMakeAgeDrop:
                if self.Customer['Inqueue_Time'][x] > self.Customer['Inqueue_Time'][minimum]:
                    minimum = x
            return self.queues.index(minimum)



    def queue_pop(self,currentTime=0):  ### Coresponds to departure
        ''' pop one customer for service
        modes = ['FCFS', 'RANDOM','LCFS','PLCFS','SJF','PSJF','SRPT','ADS','PADS']
        '''
        # check preempted customer
        if self.mode == 'FCFS':
            return self.queues.pop(0)
        if self.mode == 'LCFS' or self.mode == 'PLCFS':
            return self.queues.pop()
        if self.mode == 'LCFSE':
            returnvalue = self.queues.pop()
            self.queues = list(filter(lambda x: x > returnvalue, self.queues))
            return returnvalue
        if self.mode == 'RANDOM':
            return self.queues.pop(random.randint(0, len(self.queues) - 1))
        # if self.mode == 'RANDOME':  # comment out at 8.5.2020, since informative version is not used yet
        #     returnvalue = self.queues.pop(random.randint(0, len(self.queues) - 1))
        #     self.queues = list(filter(lambda x: x > returnvalue, self.queues))
        #     return returnvalue
        if self.mode == 'SJF' or self.mode == 'PSJF' or self.mode == 'MPSJF':
            return  self.queues.pop(self.os(len(self.queues)))
        # if self.mode=='SJFE':  # comment out at 8.5.2020, since informative version is not used yet
        #     returnvalue = self.queues.pop(self.os(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return returnvalue
        # if self.mode=='PSJFE':
        #     returnvalue= self.queues.pop(self.os(len(self.queues)))
        #     #print(returnvalue)
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return  returnvalue
        if self.mode == 'SRPT' or self.mode == 'MSRPT'or self.mode == 'SRPTA':
            return self.queues.pop(self.rs(len(self.queues)))
        # if self.mode=='AoI2PE' or self.mode=='AoI2E':    # comment out at 8.5.2020, since informative version is not used yet
        #     returnvalue = self.queues.pop(self.ageBased2_2(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return returnvalue
        # if self.mode=='AoI2':
        #     returnvalue = self.queues.pop(self.ageBased2_3(len(self.queues)))
        #     return returnvalue
        # if self.mode=='AoI3':
        #     returnvalue = self.queues.pop(self.ageBased3_4(len(self.queues)))
        #     return returnvalue
        # if self.mode=='AoI2RP' or self.mode=='AoI2R':
        #     returnvalue = self.queues.pop(self.ageBased2(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return returnvalue
        # if  self.mode=='AoI3E' or self.mode=='AoI3PE':
        #     returnvalue = self.queues.pop(self.ageBased3_3(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return returnvalue
        # if  self.mode=='AoI3R' or self.mode=='AoI3RP':
        #     returnvalue = self.queues.pop(self.ageBased3(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return returnvalue
        # if self.mode=='SRPTL':
        #     return self.queues.pop(self.rs2(len(self.queues)))
        # if self.mode=='SRPTE':
        #     returnvalue= self.queues.pop(self.rs(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return  returnvalue
        # if self.mode == 'PS':
        #     return self.queues.pop(self.rs(len(self.queues)))
        if self.mode=='TSLS':
            return self.queues.pop(self.tsls(len(self.queues)))
        if self.mode == 'ADS' or self.mode == 'PADS' or self.mode == 'MPADS' :
            return self.queues.pop(self.ageDroptoSmallest(len(self.queues), currentTime))
        # if self.mode=='ADSE' or self.mode=='PADSE':   # comment out at 8.5.2020, since informative version is not used yet
        #     returnvalue= self.queues.pop(self.ageDroptoSmallest(len(self.queues)))
        #     self.queues = list(filter(lambda x: x > returnvalue, self.queues))
        #     return returnvalue
        # if self.mode=='MPADS2':
        #     returnvalue= self.queues.pop(self.ageDroptoSmallest(len(self.queues), currentTime))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return  returnvalue
        if self.mode=='ADF' or self.mode=='PADF' or self.mode=='MPADF':
            return self.queues.pop(self.ageDropFast(len(self.queues)))
        # if self.mode=='ADFE' or self.mode=='PADFE':  # comment out at 8.5.2020, since informative version is not used yet
        #     returnvalue= self.queues.pop(self.ageDropFast(len(self.queues)))
        #     self.queues = list(filter(lambda x: x > returnvalue, self.queues))
        #     return returnvalue
        # if self.mode=='MPADF2':
        #     returnvalue= self.queues.pop(self.ageDropFast(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return  returnvalue
        if self.mode=='ADM' or self.mode=='PADM' or self.mode=='MPADM':
            return self.queues.pop(self.ageDropMost(len(self.queues)))
        # if self.mode=='MPADM2':   # comment out at 8.5.2020, since informative version is not used yet
        #     returnvalue= self.queues.pop(self.ageDropMost(len(self.queues)))
        #     self.queues=list(filter(lambda x: x>returnvalue,self.queues))
        #     return  returnvalue



    def queue_append(self, i):
        '''
        append one customer. Left ones goes out first, and right ones goes last
        modes = ['FCFS', 'RANDOM','LCFS','PS','PLCFS','FB','SJF','PSJF','SRPT','ADS]
        '''
        if self.mode in ['FCFS', 'RANDOM','RANDOME', 'LCFS','LCFSE', 'PS', 'PLCFS', 'SJF', 'SJFE','PSJF', 'SRPT','ADS','PADS','MPSJF', 'MSRPT','MPADS','PSJFE','SRPTE','MPADS2',
                         'SRPTL','SRPTA','ADF','PADF','MPADF','MPADF2','ADM','PADM','MPADM','MPADM2','AoI2PE','AoI2E','AoI3E','AoI3PE','AoI2RP','AoI2R','AoI3R','AoI3RP','ADFE',
                         'PADFE','AoI2','AoI3','PADSE','TSLS']:
            self.queues.append(i)
        else:
            print('Improper queueing mode in queue_append!', self.mode)



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


        # if self.mode == 'PS':    # comment out at 8.5.2020, since PS is not reasonable in time-slotted system
        #     if t_end == -1 or self.Customer['Remain_Work_Load'][i] <= (t_end - t_begin) / (len(self.queues) + 1):
        #         self.Customer['Serve_Intv'][i] += self.Customer['Remain_Work_Load'][i]
        #         self.Customer['Dequeue_Time'][i] = t_begin + self.Customer['Remain_Work_Load'][i] * (len(self.queues) + 1)
        #         for j in self.queues:
        #             self.Customer['Serve_Intv'][j] += self.Customer['Remain_Work_Load'][i]
        #             self.Customer['Remain_Work_Load'][j] -= self.Customer['Remain_Work_Load'][i]
        #         self.Customer['Remain_Work_Load'][i] = 0
        #         return self.depart(i)
        #     else:
        #         # self.Customer['Serve_Intv'][i] += t_end - t_begin
        #         # self.Customer['Remain_Work_Load'][i] -= t_end - t_begin
        #         self.Customer['Serve_Intv'][i] += (t_end - t_begin)/ (len(self.queues) + 1)
        #         self.Customer['Remain_Work_Load'][i] -= (t_end - t_begin) / (len(self.queues) + 1)
        #         for j in self.queues:
        #             self.Customer['Serve_Intv'][j] += (t_end - t_begin)/ (len(self.queues) + 1)
        #             self.Customer['Remain_Work_Load'][j] -= (t_end - t_begin) / (len(self.queues) + 1)
        #         return t_end


        if self.mode=='TSLS':
            if t_end == -1 or self.Customer['Remain_Work_Load'][i] <= 1:
                # customer departs
                self.Customer['Serve_Intv'][i] += self.Customer['Remain_Work_Load'][i]
                # depart time = current time + work load
                self.Customer['Dequeue_Time'][i] = t_begin + self.Customer['Remain_Work_Load'][i]
                # print(i, "-th update dequeue time: ",self.Customer['Dequeue_Time'][i])
                self.Customer['Remain_Work_Load'][i] = 0
                self.Customer['TSLS'][i] = 0
                for j in self.queues:
                    self.Customer['TSLS'][j] +=1
                return self.depart(i)
            else:
                # part of work is served
                self.Customer['Serve_Intv'][i] += 1
                self.Customer['Remain_Work_Load'][i] -= 1
                self.Customer['TSLS'][i] = 0
                for j in self.queues:
                    self.Customer['TSLS'][j] +=1
                self.queue_append(i)
                self.i_serving = -1
                return t_begin+1
        ### the following is for general case (i.e., the jobs are served one by one)
        if t_end == -1 or self.Customer['Remain_Work_Load'][i] <= t_end - t_begin:
            # customer departs
            self.Customer['Serve_Intv'][i] += self.Customer['Remain_Work_Load'][i]
            # depart time = current time + work load
            self.Customer['Dequeue_Time'][i] = t_begin + self.Customer['Remain_Work_Load'][i]
            #print(i, "-th update dequeue time: ",self.Customer['Dequeue_Time'][i])
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
            #print("age_departure:", self.Customer['Age_Dept'][i])
            self.Customer['Age_Peak'][i] = self.Customer['Age_Dept'][self.last_depart] + self.Customer['Dequeue_Intv'][
                i]
            #print("age_peak:",self.Customer['Age_Peak'][i])
            self.effe_queues.append(i)

        self.effe_departure.append(i)
        self.last_depart = i
        self.i_serving = -1

        # if self.mode=='AoI2PE' or self.mode=='AoI2E' or self.mode=='AoI3E' or self.mode=='AoI3PE':
        #     for x in self.queues:
        #         if self.Customer['Inqueue_Time'][x] < self.Customer['Inqueue_Time'][self.last_depart]:
        #             self.queues.remove(x)


        # print("departure ID: %s " % self.last_depart)
        # print("arrival time: %s " % self.Customer['Inqueue_Time'][self.last_depart])


        return self.Customer['Dequeue_Time'][i]

    def serve_between_time(self, t_begin, t_end):
        t = t_begin
        # serve the current customer
        if self.i_serving >= 0:
            t = self.serve(self.i_serving, t, t_end)
        # when there is additional time to serve other customers
        while (t < t_end or t_end == -1) and len(self.queues) > 0:
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
            return True
        elif self.mode == 'SRPT' or self.mode == 'MSRPT' or self.mode=='SRPTE' or self.mode=='SRPTL':
            return self.Customer['Remain_Work_Load'][i_new] <= self.Customer['Remain_Work_Load'][i_old]
        # elif self.mode == 'SRPTA' or self.mode=='AoI2PE' or self.mode=='AoI3PE' or self.mode=='AoI2RP' or self.mode=='AoI3RP':  # comment out at 8.5.2020, since informative version is not used yet
        #     ageAreaofPreemption=(self.Customer['Inqueue_Time'][i_old]-self.largest_inqueue_time)*(self.Customer['Remain_Work_Load'][i_new]-self.Customer['Remain_Work_Load'][i_old])
        #     ageAreaofNonPreemption=(self.Customer['Inqueue_Time'][i_new]-self.Customer['Inqueue_Time'][i_old])*self.Customer['Remain_Work_Load'][i_old]
        #     if ageAreaofPreemption<0:
        #         return True
        #     else:
        #         return ageAreaofPreemption < ageAreaofNonPreemption
        elif self.mode == 'PSJF' or self.mode == 'MPSJF' or self.mode == 'PSJFE':
            return self.Customer['Work_Load'][i_new] <= self.Customer['Work_Load'][i_old]
        elif self.mode=='PADS' or self.mode=='MPADS' or self.mode=='PADSE':
             check= self.Customer['Remain_Work_Load'][i_new] - max(self.largest_inqueue_time,self.Customer['Inqueue_Time'][i_new])<= \
                    self.Customer['Remain_Work_Load'][i_old] - max(self.largest_inqueue_time,self.Customer['Inqueue_Time'][i_old])
             return check
        elif self.mode=='PADF'or self.mode=='MPADF' or self.mode=='MPADF2' or self.mode=='PADFE':
            if self.Customer['Inqueue_Time'][i_old] < self.largest_inqueue_time:
                return True
            else:
                return  self.Customer['Remain_Work_Load'][i_new] <= self.Customer['Remain_Work_Load'][i_old]
        elif self.mode=='PADM'or self.mode=='MPADM' or self.mode=='MPADM2':
            if self.Customer['Inqueue_Time'][i_old] < self.largest_inqueue_time:
                return True
            else:
                return  self.Customer['Age_Arvl'][i_new] >= self.Customer['Inqueue_Time'][i_old]-self.largest_inqueue_time
        else:
            return False



    def preempt(self, i_old, i_new):
        '''
        i_old is preempted by i_new
        '''
        self.queue_append(i_old)
        self.i_serving=i_new
        # if self.mode == 'MSRPT'  or  self.mode == 'MPSJF' or  self.mode=='MPADS' or self.mode == 'PSJFE' or self.mode == 'SRPTE' or  \
        #         self.mode=='PADSE' or self.mode=='MPADF'  or  self.mode=='MPADF2' or self.mode=='MPADM'  or self.mode=='MPADM2' or self.mode=='AoI2PE' or self.mode=='AoI3PE':
        #     self.queues = []
        if self.mode in ['MSRPT','MPSJF','MPADS','PSJFE','SRPTE','PADSE','MPADF','MPADF2','MPADM','MPADM2','AoI2PE','AoI3PE','AoI2RP','AoI3RP','PADFE','PADSE']:
            self.queues = []




    def queueing(self):
        self.arrive(0)
        self.enqueue(0)
        # arrival index
        idx_a = 0
        # depart index
        idx_d = -1

        while idx_a < self.Nuser - 1:
            idx_a += 1
            self.serve_between_time(self.Customer['Inqueue_Time'][idx_a - 1],
                                    self.Customer['Inqueue_Time'][idx_a - 1] + self.Customer['Arrival_Intv'][idx_a])
            self.arrive(idx_a)

            if self.preemptive and self.is_preempted(self.i_serving, idx_a):
                self.preempt(self.i_serving, idx_a)
            # elif self.mode == 'PS' and self.Customer['Remain_Work_Load'][self.i_serving] > \      ## comment out at 8.5.2020, since PS is not reasonable in time-slotted system
            #         self.Customer['Remain_Work_Load'][idx_a]:
            #     self.queues.append(self.i_serving)
            #     self.i_serving = idx_a
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
                temp=self.Customer['Dequeue_Time'][self.effe_queues[x]]
                total_age += self.Customer['Age_Peak'][self.effe_queues[x]] * self.Customer['Dequeue_Time'][
                    self.effe_queues[x]] - temp*(temp-1)/2
            else:
                temp = self.Customer['Dequeue_Time'][self.effe_queues[x]] - self.Customer['Dequeue_Time'][
                    self.effe_queues[x - 1]]  # temp denotes the inter-departure time
                total_age += self.Customer['Age_Peak'][self.effe_queues[x]] * temp - temp*(temp-1)/2
            #print("total age of",x,"-th effecitve update",total_age)
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
        if self.mode in ['MSRPT','MPSJF','MPADS','PSJFE','SRPTE','PADSE','MPADF','MPADF2','MPADM','MPADM2']:
            total_response_time = 0
            for x in self.effe_departure:
                total_response_time += self.Customer['Response_Time'][x]
            return total_response_time / len(self.effe_departure)
        else:
            total_response_time = 0
            for x in range(len(self.Customer)):
                total_response_time += self.Customer['Response_Time'][x]
            return total_response_time / len(self.Customer)

    def mean_queue_len(self):
        '''
        the average queue length observed based on customer arrivals due to PASTA
        return: mean queue length
        '''
        if self.mode in ['MSRPT','MPSJF','MPADS','PSJFE','SRPTE','PADSE', 'MPADF','MPADF2','MPADM','MPADM2']:
            total_queuelength = 0
            for x in self.effe_departure:
                total_queuelength += self.Customer['Queue_Number'][x]
            return total_queuelength / len(self.effe_departure)

        else:
            return sum(self.Customer['Queue_Number'][0:] / (len(self.Customer)))

    @property
    def parameters(self):
        return 'Nuser=' + str(self.Nuser) + ', arrival_rate=' + str(self.arrival_rate) + ', mu =' + str(
            self.mu) + ', mode =' + self.mode

    # if self.Customer['Inqueue_Time'][self.queues[x]]>self.Customer['Inqueue_Time'][self.queues[y]]:
    #     if self.Customer['Inqueue_Time'][self.queues[x]]>self.Customer['Inqueue_Time'][self.queues[z]]:
    #         if self.Customer['Inqueue_Time'][self.queues[y]]>self.Customer['Inqueue_Time'][self.queues[z]]:
    #             temp1=x
    #             temp2=y
    #             temp3=z
    #         else:
    #             temp1=x
    #             temp2=z
    #             temp3=y
    #     else:
    #         temp1=z
    #         temp2=x
    #         temp3=y
    # else:
    #     if self.Customer['Inqueue_Time'][self.queues[z]]>self.Customer['Inqueue_Time'][self.queues[y]]:
    #         temp1 = z
    #         temp2 = y
    #         temp3 = x
    #     else:
    #         if self.Customer['Inqueue_Time'][self.queues[x]]>self.Customer['Inqueue_Time'][self.queues[z]]:
    #             temp1 = y, temp2 = x, temp3 = z
    #         else:
    #             temp1 = y
    #             temp2 = z
    #             temp3 = x