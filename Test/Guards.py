#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 09:13:26 2021

@author: emran
"""
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


class Guard:

    def __init__(self, hardConstraintPenalty):

        self.hardConstraintPenalty = hardConstraintPenalty

        # list of guards:
        self.guards = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        self.shiftPreference = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]]

        self.shiftMin = [2, 2, 1]
        self.shiftMax = [3, 4, 2]

        # max shifts per week allowed for each guard
        self.maxShiftsPerWeek = 5

        # number of weeks we create a schedule for:
        self.weeks = 1

        # useful values:
        self.shiftPerDay = len(self.shiftMin)
        self.shiftsPerWeek = 7 * self.shiftPerDay

    def __len__(self):

        return len(self.guards) * self.shiftsPerWeek * self.weeks

    def getCost(self, schedule):

        guardShiftsDict = self.getGuardShifts(schedule)

        consecutiveShiftViolations = self.countConsecutiveShiftViolations(guardShiftsDict)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(guardShiftsDict)[1]
        guardsPerShiftViolations = self.countGuardsPerShiftViolations(guardShiftsDict)[1]
        shiftPreferenceViolations = self.countShiftPreferenceViolations(guardShiftsDict)

        hardContstraintViolations = consecutiveShiftViolations + guardsPerShiftViolations + shiftsPerWeekViolations
        softContstraintViolations = shiftPreferenceViolations

        return self.hardConstraintPenalty * hardContstraintViolations + softContstraintViolations

    def getGuardShifts(self, schedule):

        shiftsPerGuard = self.__len__() // len(self.guards)
        guardShiftsDict = {}
        shiftIndex = 0

        for guard in self.guards:
            guardShiftsDict[guard] = schedule[shiftIndex:shiftIndex + shiftsPerGuard]
            shiftIndex += shiftsPerGuard

        return guardShiftsDict

    def countConsecutiveShiftViolations(self, guardShiftsDict):

        violations = 0
        # iterate over the shifts of each guard:
        for guardShifts in guardShiftsDict.values():
            # look for two cosecutive '1's:
            for shift1, shift2 in zip(guardShifts, guardShifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations

    def countShiftsPerWeekViolations(self, guardShiftsDict):

        violations = 0
        weeklyShiftsList = []
        # iterate over the shifts of each guard:
        for guardShifts in guardShiftsDict.values():  # all shifts of a single guard
            # iterate over the shifts of each weeks:
            for i in range(0, self.weeks * self.shiftsPerWeek, self.shiftsPerWeek):
                # count all the '1's over the week:
                weeklyShifts = sum(guardShifts[i:i + self.shiftsPerWeek])
                weeklyShiftsList.append(weeklyShifts)
                if weeklyShifts > self.maxShiftsPerWeek:
                    violations += weeklyShifts - self.maxShiftsPerWeek

        return weeklyShiftsList, violations

    def countGuardsPerShiftViolations(self, guardShiftsDict):

        totalPerShiftList = [sum(shift) for shift in zip(*guardShiftsDict.values())]

        violations = 0
        # iterate over all shifts and count violations:
        for shiftIndex, numOfGuards in enumerate(totalPerShiftList):
            dailyShiftIndex = shiftIndex % self.shiftPerDay  # -> 0, 1, or 2 for the 3 shifts per day
            if (numOfGuards > self.shiftMax[dailyShiftIndex]):
                violations += numOfGuards - self.shiftMax[dailyShiftIndex]
            elif (numOfGuards < self.shiftMin[dailyShiftIndex]):
                violations += self.shiftMin[dailyShiftIndex] - numOfGuards

        return totalPerShiftList, violations

    def countShiftPreferenceViolations(self, guardShiftsDict):

        violations = 0
        for guardIndex, shiftPreference in enumerate(self.shiftPreference):
            # duplicate the shift-preference over the days of the period
            preference = shiftPreference * (self.shiftsPerWeek // self.shiftPerDay)
            # iterate over the shifts and compare to preferences:
            shifts = guardShiftsDict[self.guards[guardIndex]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1

        return violations

    def printScheduleInfo(self, schedule):

        guardShiftsDict = self.getGuardShifts(schedule)

        print("Schedule for each guard:")
        for guard in guardShiftsDict:  # all shifts of a single guard
            print(guard, ":", guardShiftsDict[guard])

        print("consecutive shift violations = ", self.countConsecutiveShiftViolations(guardShiftsDict))
        print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(guardShiftsDict)
        print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()

        totalPerShiftList, violations = self.countGuardsPerShiftViolations(guardShiftsDict)
        print("Guards Per Shift = ", totalPerShiftList)
        print("Guards Per Shift Violations = ", violations)
        print()

        shiftPreferenceViolations = self.countShiftPreferenceViolations(guardShiftsDict)
        print("Shift Preference Violations = ", shiftPreferenceViolations)
        print()

# In[3]:
