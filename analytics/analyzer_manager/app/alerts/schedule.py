import enum
from dataclasses import dataclass
from typing import Dict, Optional
from general import apply_from_dict
import time

ALLDAY = 'AllDay'
# from 12 hour to 24-hour format
# Function to convert the date format
def convert24(str1, conver_24 = False):
    h = int(str1.split(":")[0])
    m = int(str1.split(":")[1].split()[0])
    if str1[-2:] == "AM" and str1[:2] == "12":
        if conver_24:
            return 24, m
        else:
            return 0, m


    # remove the AM
    if str1[-2:] == "AM":
        return h, m
    # checking if last two elements of time
    # is PM and first two elements are 12
    if str1[-2:] == "PM" and str1[:2] == "12":
        return h, m
    # add 12 to hours and remove PM
    return (h + 12), m


class DayName(int, enum.Enum):
    monday = 0
    tuesday = 1
    wednesday = 2
    thursday = 3
    friday = 4
    saturday = 5
    sunday = 6

class ArmMode(int, enum.Enum):
    Arm = 0
    Disarm = 1
    Schedule = 2

@dataclass
class HourMinute:
    hour: int
    minute: int


class Schedule:
    _schedule: Dict[str, Optional[Dict[str, HourMinute]]]

    def __init__(self, schedule=None):
        self._all_week: bool = True
        self._schedule = {}
        if not (type(schedule) is list or type(schedule) is dict):
            schedule = [
                {"day":0,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},
                {"day":1,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},
                {"day":2,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},
                {"day":3,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},
                {"day":4,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},
                {"day":5,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]},
                {"day":6,"enabled":True,"times":[{"from":"12:00 AM","to":"12:00 AM"}]}
                ]               
            
        self._all_week = True
        for day_dict in schedule:
            if day_dict is not None:
                day = DayName(int(day_dict["day"])).name
                if "times" in day_dict and type(day_dict["times"]) is list:
                        if not(day_dict["enabled"]) or (len(day_dict["times"]) == 0):
                            self._all_week = False
                            self._schedule[day] = None
                        else: 
                            self._schedule[day] = []                            
                            # running on each element and checking
                            for time in day_dict["times"]:                                            
                                # translate text to  hours
                                fh, fm = convert24(time["from"])
                                eh, em = convert24(time["to"], conver_24=True)
                                #check if its all day
                                if fh == 0 and fm == 0 and (
                                    (eh == 0 and em == 0) or
                                    (eh == 24 and em == 0)
                                ):
                                    self._schedule[day] = ALLDAY
                                    break
                                else:
                                    self._all_week = False
                                    self._schedule[day].append(
                                        {"from": HourMinute(fh, fm), "to": HourMinute(eh, em)}
                                    )                                
                                
                else:
                    if not (day_dict["allDay"]):
                        self._all_week = False                    
                        fh, fm = convert24(day_dict["from"])
                        eh, em = convert24(day_dict["to"], conver_24=True)
                        self._schedule[day] = [{"from": HourMinute(fh, fm), "to": HourMinute(eh, em)}]  
                    else:
                        self._schedule[day] = ALLDAY
        

        



    def check_schedule(self, now, day):        
        if self._all_week:
            return True
        
        # if we got here schedule is defined
        if self._schedule[day] is None:
            return False
        
        if self._schedule[day] == ALLDAY:
            return True        

        # if we got here, hours are defined
        h = now.hour
        m = now.minute

        # running on each timeslot in the day
        
        for time in self._schedule[day]:            
            if (
                h > time["from"].hour
                or ((h == time["from"].hour) and (m >= time["from"].minute))
            ) and (
                h < time["to"].hour
                or ((h == time["to"].hour) and (m <= time["to"].minute))
            ):
                return True
        
        # if we got here - out of schedule
        return False

    @property
    def all_week(self):
        return self._all_week

class Policy:
    def __init__(self, policy_msg=None):
        self.mode = apply_from_dict("mode",policy_msg,ArmMode.Schedule.value)
        schedule = apply_from_dict("schedule",policy_msg,None)
        self.schedule = Schedule(schedule)        
        self.init_time = apply_from_dict("timestamp",policy_msg,time.time()*1000)
        self.auto_rearm = apply_from_dict("autoRearm",policy_msg,0)
        self.auto_disarm = apply_from_dict("autoDisarm",policy_msg,0)
    

    def check_policy(self, now, day):
        current_time = time.time()*1000
        if (self.mode == ArmMode.Disarm.value):
            if self.auto_rearm and ((current_time - self.init_time)/60000 > self.auto_rearm):
                self.mode = ArmMode.Schedule.value                            
            else:
                return False
        
        if (self.mode == ArmMode.Arm.value):
            if self.auto_disarm and ((current_time - self.init_time)/60000 > self.auto_disarm):
                self.mode = ArmMode.Schedule.value
            else:
                return True                                     
        
        #Check schedule
        return self.schedule.check_schedule(now, day)               

    def all_time_schedule(self):
        return self.schedule.all_week


class PolicyV2:
    def __init__(self, policy_msg=None):
        current_time = time.time()*1000

        self.mode = apply_from_dict("mode",policy_msg,ArmMode.Arm.value)
        self.schedule = Schedule(apply_from_dict("schedule",policy_msg,None))
        self.lastManualOp = apply_from_dict("lastManualOp",policy_msg,None)       
        self.auto_rearm = apply_from_dict("autoRearm",policy_msg,0)
        self.auto_disarm = apply_from_dict("autoDisarm",policy_msg,0)        

    def update(self, policy_msg):
        current_time = time.time()*1000
        new_mode = apply_from_dict("mode",policy_msg,self.mode)
        self.lastManualOp = apply_from_dict("lastManualOp",policy_msg,None)
        self.schedule = Schedule(apply_from_dict("schedule",policy_msg,None))        
        self.auto_rearm = apply_from_dict("autoRearm",policy_msg,self.auto_rearm)
        self.auto_disarm = apply_from_dict("autoDisarm",policy_msg,self.auto_disarm)
        
        if (new_mode != self.mode):
            #mode update
            if not(self.lastManualOp):
                #we didnt get value from the cloud
                self.lastManualOp = current_time
        self.mode = new_mode

    def check_policy(self, now, day):
        current_time = time.time()*1000
        if (self.mode == ArmMode.Disarm.value):            
            if self.auto_rearm:
                # we have auto rearm
                if self.lastManualOp:
                    # its not disarm from init
                    if ((current_time - self.lastManualOp)/60000 > self.auto_rearm):
                        # we passed the disarm time
                        self.mode = ArmMode.Arm.value
                        # last manual operation recovered
                        self.lastManualOp = None
                        return self.schedule.check_schedule(now, day)
                    else:
                        # we are still in disarm time
                        return False
                else:
                    # On init there is no reArm
                    return False

            else:
                # no auto rearm, we are disarmed until we get enter arm schedule
                if self.schedule.all_week:
                    return False
                if self.schedule.check_schedule(now, day):
                    self.mode = ArmMode.Arm.value
                    return True
                else:
                    return False
        else:
            if self.auto_disarm:
                # we have auto disarm
                if self.lastManualOp:
                    # its not arm from init
                    if ((current_time - self.lastManualOp)/60000 > self.auto_disarm):
                        # we passed the arm time
                        self.mode = ArmMode.Disarm.value
                        self.lastManualOp = None
                        return False
                    else:
                        # we are still in arm time
                        return self.schedule.check_schedule(now, day)
                else:
                    # On init there is no reArm
                    return self.schedule.check_schedule(now, day)

            else:
                # no auto disarm, we are armed until we get enter disarm schedule
                if self.schedule.all_week:
                    return True
                if self.schedule.check_schedule(now, day):
                    return True
                else:
                    self.mode = ArmMode.Disarm.value
                    return False          

    def all_time_schedule(self):
        return self.schedule.all_week
        

class ZonePolicy:
    def __init__(self, zone_msg=None):
        self.partitionPolicy = PolicyV2(apply_from_dict("partitionPolicy",zone_msg,{}))
        self.accountPolicy = PolicyV2(apply_from_dict("accountPolicy",zone_msg,{}))    
    
    def update(self, zone_msg):
        self.partitionPolicy.update(apply_from_dict("partitionPolicy",zone_msg,{}))
        self.accountPolicy.update(apply_from_dict("accountPolicy",zone_msg,{}))

    def check_zone(self, now, day):
        # We are getting here cause alert is all time
        if self.partitionPolicy.check_policy(now, day):
            #control zone is armed
            if self.partitionPolicy.all_time_schedule():
                #since the control zone is with all week schedule - we need to check zone policy
                return self.accountPolicy.check_policy(now, day)
            else:
                # control zone is armed with schedule, stopping here
                return True
        else:
            # control zone is disarmed, stopping here
            return False    
        
           
