from argparse import Namespace
from typing import Dict, Optional
import copy
from general.core import ClassHandler, DashboardsAction
from typing import Dict, Optional
from typing import Dict, Any, Optional

from general.analyzer_general import inRoi, roi_gen, getTrafficPath, POSITION_LIST, SUPPORTED_FILTERS
import statistics
from alerts.alerts import AlertsAction
import enum

# Message acknolage sent when all models are up
class e_VariableType(enum.IntEnum):
    OBJECT = (0,)
    EVENT = (1,)

class VariablesGroup(object):
    category: str = "base"

    def __init__(self,analyticConfig):

        
        self.lastMinuteUpdate = 0
        self.variables = {}
        self._analytic_config = analyticConfig
             
        
    
    def addVariable(self, variable):
        #object flow
        if variable.object_based_variable:
            if not(variable.object in self.variables): self.variables[variable.object] = {}
            self.variables[variable.object][variable.variableId] = variable
        else:
            if not "events" in self.variables: self.variables["events"] = {}
            self.variables["events"][variable.variableId] = variable

    def removeVariable(self, variableId):
        remove = False
        for type in self.variables:
            for index in self.variables[type]:
                if variableId == self.variables[type][index].variableId:
                    remove = True
                    t_type = type
                    t_index = index
                    break
        if remove: 
            del self.variables[t_type][t_index]
            if not(len(self.variables[t_type])):
                del self.variables[t_type]
    
    def matchFilters(self, variable, desc):
        if (variable.filtersDisabled):
            return True
        
        for key in variable.filters.keys():
            a_match = True
            # if we dont have such attribute we still enable match
            if len(variable.filters[key]) and key in desc:
                p_match = False
                for parameter in variable.filters[key]:
                    # check if attribute is a string
                    if type(desc[key]) == str:
                        if parameter == desc[key]:
                            p_match = True
                            break
                    # if its a list/dic look if there is at least one match
                    else:
                        if parameter in desc[key]:
                            # We found match
                            p_match = True
                            break

                a_match = a_match and p_match

            if not (a_match):
                return False
        # we matched on all attributes
        return True
        
    
    def update_object_count(self,id,object,ppargs):
        idIndex = object["idIndex"]
        type = ppargs.attributes[idIndex]["type"]
        self.variables[type][id].frameInfo.update = True
                
        self.variables[type][id].frameInfo.count +=1        
        self.variables[type][id].statistic.appearance[idIndex] = True
    
    
    def update_heatmaps(self, frame_item):
        if "info" in frame_item:
            for object in frame_item["info"]:
                
                #run only on tracked objects
                if not("idIndex" in object):
                    continue                
                
                for type in self.variables:
                    if not (type=="events"):
                        self.update_object_heatmap(type,object)                           

    
    def update_object_heatmap(self, type, object):
        for id in self.variables[type]:
            if object["idIndex"] in self.variables[type][id].statistic.appearance:                
                for position in object["markedPositions"]:
                    period = position["endTime"] - position["startTime"]
                    self.variables[type][id].statistic.heatmap[position["index"]] += period
                    # if period: 
                    #     if position["index"] in self.variables[type][id].frameInfo.heatmap:
                    #         self.variables[type][id].statistic.heatmap[position["index"]] += period
                    #     else:
                    #         self.variables[type][id].statistic.heatmap[position["index"]] =  period        
    
    def update_object_trafficpath(self, type, idIndex, markedPositions):
        if type in self.variables:
            for id in self.variables[type]:
                if idIndex in self.variables[type][id].statistic.appearance:                
                    #we need to update the trafficPaths
                    newPath = getTrafficPath(markedPositions,self._analytic_config["variableConfig"]["trafficPathResolution"])
                    #go over existing traffic paths
                    new = True
                    for trafficPath in self.variables[type][id].statistic.trafficPath:
                        if trafficPath["path"] == newPath:
                            trafficPath["count"] += 1
                            new = False
                            break
                    
                    #if its a new path just update
                    if new: self.variables[type][id].statistic.trafficPath.append({"count": 1,"path": newPath})      

                

    
    def summary_frameInfo(self):
        realTimeVariables = []
        for type in self.variables:
            for id in self.variables[type]:  
                #build object based info
                if self.variables[type][id].object_based_variable:              
                    if self.variables[type][id].frameInfo.update:
                        #update heatmap
                        # for position in self.variables[type][id].frameInfo.heatmap:
                        #     if position in self.variables[type][id].statistic.heatmap:
                        #         self.variables[type][id].statistic.heatmap[position] += self.variables[type][id].frameInfo.heatmap[position]
                        #     else:
                        #         self.variables[type][id].statistic.heatmap[position]  = self.variables[type][id].frameInfo.heatmap[position]
                        for idx in range(len(self.variables[type][id].frameInfo.heatmap)):
                            self.variables[type][id].statistic.heatmap[idx] += self.variables[type][id].frameInfo.heatmap[idx]

                        #update total count value
                        self.variables[type][id].statistic.count.append(self.variables[type][id].frameInfo.count)
                        self.variables[type][id].frameInfo = copy.deepcopy(Namespace(count = 0, trafficPath = [], heatmap = POSITION_LIST, update = False))
                    
                    else:
                        #no appearance
                        self.variables[type][id].statistic.count.append(0)
                    
                    #on real time variables check if we had a change
                    if self.variables[type][id].realTime and len(self.variables[type][id].statistic.count) > 2:
                        if not(self.variables[type][id].statistic.count[-1] == self.variables[type][id].statistic.count[-1]):
                            realTimeVariables.append(self.buildStatistic(type,id,periodUpdate=False))
        return realTimeVariables

                
                
    
    def update_variables(self, ppargs, frame_item, variablesEvents):
        realTimeVariables = []

        if "info" in frame_item:
            for object in frame_item["info"]:
                
                #run only on tracked objects
                if not("idIndex" in object):
                    continue
                
                #run only on variables that are in the scene 
                if (object["position"]==None):
                    continue

                idIndex = object["idIndex"]                                
                type = ppargs.attributes[idIndex]["type"]
                
                if not(type in self.variables):
                    continue                
                
                for id in self.variables[type]:
                    if self.matchFilters(self.variables[type][id],ppargs.attributes[idIndex]["description"]) and inRoi(self.variables[type][id].roiSettings, object["position"]):  
                        self.update_object_count(id,object,ppargs)
                
        if "events" in self.variables:
            for variableId in self.variables["events"]:
                if variableId in variablesEvents:                    
                    self.variables["events"][variableId].statistic.count +=1
                    self.variables["events"][variableId].statistic.data.append(variablesEvents[variableId]["alertData"])
                    if self.variables["events"][variableId].realTime:
                        realTimeVariables.append(self.buildStatistic("events",variableId,periodUpdate=False))

        realTimeVariables.extend(self.summary_frameInfo())
        return realTimeVariables
    
    
    def buildStatistic(self,type,id,periodUpdate = True):
        variable = {}
        variable["variableId"]      = self.variables[type][id].variableId
        variable["variableType"]    = self.variables[type][id].variableType
        variable["periodInMinutes"] = self.variables[type][id].periodInMinutes

        if periodUpdate: variable["realTime"] = False
        else:            variable["realTime"] = True

        if self.variables[type][id].object_based_variable:
            variable["count"] = 0
            for tracker in self.variables[type][id].statistic.appearance:
                if not(tracker in self.variables[type][id].statistic.prevAppearance):
                    variable["count"] += 1

            variable["stat"] = {}
            variable["stat"]["avg"]    = round(statistics.mean(self.variables[type][id].statistic.count),2)
            variable["stat"]["min"]    = min(self.variables[type][id].statistic.count)
            variable["stat"]["max"]    = max(self.variables[type][id].statistic.count)
            variable["stat"]["latest"] = self.variables[type][id].statistic.count[-1]

            if periodUpdate:
                variable["heatmap"] = []
                # for index in self.variables[type][id].statistic.heatmap:
                #     periodInSeconds = round(self.variables[type][id].statistic.heatmap[index] / 1000,1)                                       
                #     variable["heatmap"].append({"index": index, "period": periodInSeconds})
                for idx in range(len(self.variables[type][id].statistic.heatmap)):
                    if not(len(self.variables[type][id].markedIdx)) or (idx in self.variables[type][id].markedIdx):
                        variable["heatmap"].append(round(self.variables[type][id].statistic.heatmap[idx] / 1000,1))
                    else:
                        variable["heatmap"].append(0)

                variable["trafficPath"] = self.variables[type][id].statistic.trafficPath
        else:
            variable["count"] = self.variables[type][id].statistic.count
            statisticData = list(filter(lambda item: item is not None, self.variables[type][id].statistic.data))
            if len(statisticData):
                variable["stat"]        = {}
                variable["stat"]["avg"]    = round(statistics.mean(statisticData),2)
                variable["stat"]["min"]    = min(statisticData)
                variable["stat"]["max"]    = max(statisticData)
                variable["stat"]["latest"] = statisticData[-1]                   
        return variable

    def dumpStatistic(self, minute):       
        self.lastMinuteUpdate = minute
        variables = []
        
        for type in self.variables:
            for id in self.variables[type]:
                variable = self.buildStatistic(type,id,periodUpdate=True)
                
                #reset statistic
                if self.variables[type][id].object_based_variable:                  
                    self.variables[type][id].statistic = copy.deepcopy(Namespace(appearance = {}, prevAppearance = self.variables[type][id].statistic.appearance, count = [], trafficPath = [], heatmap = POSITION_LIST))     
                else:                                       
                    self.variables[type][id].statistic = copy.deepcopy(Namespace(count = 0, data = []))
                
                variables.append(variable)

        return variables
    
    @staticmethod
    def _apply_from_dict(key, alert_dict, default_value):
        if key in alert_dict and not (alert_dict[key] == None):
            return alert_dict[key]
        else:
            return default_value


class DashboardManager:
    _groups: Dict[str, VariablesGroup]
    _analytic_config: Dict
    _class_handler: ClassHandler
        

    def __init__(self, init_dict, config, class_handler, alert_manager):
        self._groups = {}
        self._analytic_config = config
        self._class_handler = class_handler
        self.alert_manager = alert_manager  

        if "variables" in init_dict:
            for variable_dict in init_dict["variables"]:
                self._add_variable(variable_dict)

    def get_class_handler(self):
        return self._class_handler

    def get_config(self):
        return self._analytic_config
    
    def generate_variableGroup(self) -> VariablesGroup:
        return VariablesGroup(self._analytic_config)
    
    def getVariableId(self, variable_dict: Dict = {}):
            variableId: str = self._apply_from_dict("variableId", variable_dict, "")
            return variableId
    
    def parseVariableEvent(self,variable_dict: Dict = {}, add=True):
        #build alert like message
        alert = variable_dict["event"]
        if add: alert["action"] = AlertsAction.add.value
        else:   alert["action"] = AlertsAction.remove.value
        alert["_id"] = variable_dict["_id"]
        alert["enabled"] = True
        alert["variableAlert"] = True
        self.alert_manager.parse_alerts_msg({"alert": alert})

    def generate_variable(self, variable_dict: Dict = {},add=True):
            variableId: str = self._apply_from_dict("_id", variable_dict, "")
            periodInMinutes  = self._apply_from_dict("periodInMinutes", variable_dict, 5)
            periodInMinutes  = max(periodInMinutes,1)
            realTime         = self._apply_from_dict("realTime", variable_dict, False)
            

            if "configuration" in variable_dict and variable_dict["configuration"] is not None:
                #look for roi settings
                configuration_dict = variable_dict["configuration"]            
                markedIdx = self._apply_from_dict("markedIdx", configuration_dict, [])
                if markedIdx is None: markedIdx = []
                roiFilter: bool = len(markedIdx) > 0
                roi: Any = None if not roiFilter else roi_gen(markedIdx)
                variableType = e_VariableType.OBJECT.value

                roiSettings = copy.deepcopy(Namespace(roi = roi, roiFilter = roiFilter))

                object_value = self._apply_from_dict("object", configuration_dict, None)
                
                if object_value is not None:
                    object = self.get_class_handler().object_int_to_str(object_value)
                else:
                    object = "unknown"

                m_filters: Optional[object] = self._apply_from_dict("filters", configuration_dict, {})
                filters = {}
                for key in m_filters.keys():
                    if key in SUPPORTED_FILTERS:
                        filters[key] = m_filters[key]
                #check if filters are set
                filtersDisabled = True
                for filter in filters:
                    if type(filters[filter]) == list and len(filters[filter]): filtersDisabled = False
                
                frameInfo = copy.deepcopy(Namespace(count = 0, trafficPath = [], heatmap = POSITION_LIST, update = False))
                statistic = copy.deepcopy(Namespace(appearance = {}, prevAppearance = {}, count = [], trafficPath = [], heatmap = POSITION_LIST))
                return copy.deepcopy(Namespace(object_based_variable = True, variableType = variableType, periodInMinutes = periodInMinutes, realTime = realTime, object = object, variableId = variableId, roiSettings = roiSettings, filters = filters, filtersDisabled = filtersDisabled, markedIdx = markedIdx, statistic = statistic, frameInfo = frameInfo))
                
            elif "event" in variable_dict and variable_dict["event"] is not None:
                self.parseVariableEvent(variable_dict,add)
                variableType = e_VariableType.EVENT.value
                statistic = copy.deepcopy(Namespace(count = 0, data = []))
                return copy.deepcopy(Namespace(object_based_variable = False, variableType = variableType, periodInMinutes = periodInMinutes, realTime = realTime, variableId = variableId, statistic = statistic))

            return None                            
    
    @staticmethod
    def _apply_from_dict(key, alert_dict, default_value):
        if key in alert_dict and not (alert_dict[key] == None):
            return alert_dict[key]
        else:
            return default_value
    

    def deleteEventVariable(self, variableId):
        delete = False
        for dashborad_id in self._groups:
            if "events" in self._groups[dashborad_id].variables and variableId in self._groups[dashborad_id].variables["events"]:
                delete = True
                del self._groups[dashborad_id].variables["events"][variableId]
                if len(self._groups[dashborad_id].variables["events"]) == 0:
                    del self._groups[dashborad_id].variables["events"]                    
        return delete
    
    def _add_variable(self, variable_dict):
        new_variable = self.generate_variable(variable_dict, add = True)
        if new_variable is not None:
            if not new_variable.periodInMinutes in self._groups: self._groups[new_variable.periodInMinutes] = self.generate_variableGroup()
            self._groups[new_variable.periodInMinutes].addVariable(new_variable)

    def _remove_variable(self, variable_dict):
        variableId = self.getVariableId(variable_dict)
        self.generate_variable(variable_dict, add = False)
        for group in self._groups:
            self._groups[group].removeVariable(variableId)
        
    
    def parse_dashboard_msg(self, dashboard_msg):
        if "variable" in dashboard_msg:
            variable_dict = dashboard_msg["variable"]
            action = DashboardsAction(variable_dict["action"])
            if action is DashboardsAction.add:
                self._add_variable(variable_dict)
            elif action is DashboardsAction.remove:
                self._remove_variable(variable_dict)
            elif action is DashboardsAction.update:
                self._remove_variable(variable_dict)
                self._add_variable(variable_dict)
