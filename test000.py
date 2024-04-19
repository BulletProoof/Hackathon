# 环境设定
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, creator, algorithms
import random
from copy import deepcopy
from pprint import pprint
import datetime

# 自编写功能模块
from getData import get_data
from DataFormat import *

node_dict, vehicle_dict, cost_dict, order_dict = get_data()

# TODO1：路径优化，合并起点，一起送到终点。
# TODO2：车辆载重会随着动态变化，并不是全加一起，这里很关键
# TODO3：optimizePermutation.py


params = {
    'font.family': 'serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'legend.fontsize': 'small'
}
plt.rcParams.update(params)

# -----------------------------------
## 问题定义
creator.create('FitnessMin', base.Fitness, weights=(-1.0,))  # 最小化问题
# 给个体一个routes属性用来记录其表示的路线
creator.create('Individual', list, fitness=creator.FitnessMin)

# -----------------------------------
# 个体编码
# 用字典存储所有参数 -- 配送中心坐标、顾客坐标、顾客需求、到达时间窗口、服务时间、车型载重量
dataDict = {}
# 节点坐标，节点0是配送中心的坐标
dataDict['NodeCoor'] = [(15, 12), (3, 13), (3, 17), (6, 18), (8, 17), (10, 14),
                        (14, 13), (15, 11), (15, 15), (17, 11), (17, 16),
                        (18, 19), (19, 9), (19, 21), (21, 22), (23, 9),
                        (23, 22), (24, 11), (27, 21), (26, 6), (26, 9),
                        (27, 2), (27, 4), (27, 17), (28, 7), (29, 14),
                        (29, 18), (30, 1), (30, 8), (30, 15), (30, 17)
                        ]
# 将配送中心的需求设置为0
dataDict['Demand'] = [0, 50, 50, 60, 30, 90, 10, 20, 10, 30, 20, 30, 10, 10, 10,
                      40, 51, 20, 20, 20, 30, 30, 30, 10, 60, 30, 20, 30, 40, 20, 20]
# 将配送中心的服务时间设置为0
dataDict['Timewindow'] = [(0, 0), (3, 4), (4, 5), (5, 7), (2, 9), (10, 11),
                          (12, 13), (11, 13), (1, 5), (11, 15), (8, 9),
                          (10, 12), (15, 16), (18, 20), (1, 7), (6, 9),
                          (8, 9), (4, 6), (7, 11), (10, 15), (13, 14),
                          (17, 18), (16, 18), (10, 12), (13, 14), (5, 8),
                          (6, 8), (7, 9), (3, 6), (4, 5), (6, 9)
                          ]
dataDict['MaxLoad'] = 400
dataDict['ServiceTime'] = 1
dataDict['Velocity'] = 30  # 车辆的平均行驶速度


def genInd():
    """
    根据订单数据（4251条历史数据）生成初始的运输策略，包括：每辆车的车型、车辆服务顾客、先后顺序、
    编码方式： '实数编码'，   e.g.  4.567567  中的 4表示车型4,0.567567作为优先级，决定顾客的顺序
    暂时不考虑订单拆分和订单合并
    :param :
    :return:
    """
    """生成个体， 对我们的问题来说，困难之处在于车辆数目是不定的"""

    n_order = len(order_dict)  # 订单数量
    # 生成订单的随机数数组，范围从0-5，因为车型只有0,1,2,3,4这5种。
    # 考虑第一次生成可行方案的时候，需要每辆车都大于最大的volume，52。故，选择2号车型开始随机
    # ind = np.random.uniform(0, 5, n_order)
    ind = np.random.uniform(2, 5, n_order)

    # 先将同车型的订单放到一起排序，初始化5个数组，存放5种不同车型的顾客
    load_arr = [[] for _ in range(5)]
    for idx, num in enumerate(ind):
        load_arr[abs(int(num))].append((idx + 1, num))  # 注意order_id是从1开始的

    #  在排列中，负数作为标志，定位某一车辆的运输路线最后一个顾客
    for i in range(5):
        pointer = 0  # 迭代指针
        lowPointer = 0  # 指针指向下界
        max_load = vehicle_dict[i].upper
        # max_load = vehicle_dict[i][1]
        length = len(load_arr[i])
        if length == 0:
            continue
        # 开始生成每种车型0,1,2,3,4的路线
        # 当指针不指向序列末尾时
        while pointer <= length - 1:
            vehicleLoad = 0.0
            # 当不超载时，继续装载，现在只考虑上界，暂时不考虑下界
            while (pointer <= length - 1) and (vehicleLoad <= max_load):
                order_id = load_arr[i][pointer][0]
                volume = order_dict[order_id].volume  # volume, max(volume=52)
                vehicleLoad += volume
                pointer += 1
            # 在第一次生成可行方案的时候，保证不会出现，顾客的需求大于车容量，也就是说不会出现lowPointer+1>=pointer的情况
            # if lowPointer + 1 < pointer:
            tempPointer = np.random.randint(lowPointer, pointer)
            order_id = load_arr[i][tempPointer][0]
            ind[order_id - 1] *= -1  # 置为负数，作为一辆车的最后顾客的标识，order_id从1开始的
            lowPointer = tempPointer + 1
            pointer = lowPointer

    # 将路线片段合并为染色体
    return ind


# -----------------------------------
# 评价函数
# 染色体解码
def decodeInd(ind):
    """从染色体解码回路线片段"""
    customers = defaultdict(list)
    routes = defaultdict(list)
    for i in range(5):
        route_slice = []
        for j in range(len(ind)):
            num = ind[j]
            idx = j + 1
            if abs(int(num)) == i:
                route_slice.append(idx)
                if num < 0.0:
                    customers[i].append(route_slice)
                    route_slice = []
            if j == len(ind) - 1 and len(route_slice) != 0:
                customers[i].append(route_slice)
                route_slice = []
    for key, value in customers.items():
        route_slice = []
        for i in range(len(value)):
            temp_lst = []
            lst = value[i]
            for it in lst:
                origin_id = order_dict[it].origin_id
                destination_id = order_dict[it].destination_id
                temp_lst.append(origin_id)
                temp_lst.append(destination_id)
            route_slice.append(temp_lst)
        routes[key] = route_slice
    return customers, routes


def cal_route_cost(route_dict):
    """
    计算染色体解码路径的运输费用
    :param route_dict:
    :return:
    """
    routes_cost = 0.0
    for key, lists in route_dict.items():
        for lst in lists:
            single_route = 0.0
            single_duration = 0
            for x, y in zip(lst[0:], lst[1:]):
                # 判断cost_dict是否有（x,y)对于的Cost对象
                if (x, y) in cost_dict.keys():
                    single_route += cost_dict[(x, y)].cost
                elif x == y:
                    single_route += 0.0
                else:
                    print('no find')
                    single_route += INF
                routes_cost += single_route
    return routes_cost

def cal_total_route_cost(route_dict):
    """
    计算染色体解码路径的运输费用+固定费用，最小化目标之一
    :param route_dict:
    :return:
    """
    total_routes_cost = 0.0
    for key, lists in route_dict.items():
        total_routes_cost += vehicle_dict[key].price * len(lists)
        for lst in lists:
            single_route = 0.0
            single_duration = 0
            for x, y in zip(lst[0:], lst[1:]):
                # 判断cost_dict是否有（x,y)对于的Cost对象
                if (x, y) in cost_dict.keys():
                    single_route += cost_dict[(x, y)].cost
                elif x == y:
                    single_route += 0.0
                else:
                    print('no find')
                    single_route += INF
                total_routes_cost += single_route
    return total_routes_cost

def cal_route_waitting_time(schedule_dict):
    """
    计算所有订单等待时间，最小化目标之一
    :param schedule_dict:
    :return:
    """
    waitting_time = 0
    for k, v in schedule_dict.items():
        # time_diff类型为：datetime
        time_diff = v.actual_pickup_time - v.expect_pickup_time
        waitting_time += time_diff.total_seconds()
    return waitting_time

def cal_route_load(customers):
    """
    计算染色体解码的路线载重
    :param customers:各种车型的路线字典，例子：{2: [[1, 4], [5, 6]],3:[8, 12], [13, 16, 17]]}
    :return:
    """
    route_load = 0.0
    for key, val in customers.items():
        for each_route in val:
            route_load += np.sum([order_dict[i].volume for i in each_route])
    return route_load

def loadPenalty(customers):
    """
    辅助函数，因为在交叉和突变中可能会产生不符合负载约束的个体，需要对不合要求的个体进行惩罚
    计算每条路径的负载，取max(0, routeLoad - maxLoad)计入惩罚项
    :param customers: 各种车型的路线字典，例子：{2: [[1, 4], [5, 6]],3:[8, 12], [13, 16, 17]]}
    :return: penalty 惩罚系数
    """
    penalty = 0
    # key为车型，val对应的多条路线
    for key, val in customers.items():
        for each_route in val:
            each_route_load = np.sum([order_dict[i].volume for i in each_route])
            penalty += max(0, each_route_load - vehicle_dict[key].upper)
    return penalty


def calcRouteServiceTime(customers, routes):
    """
    根据染色体，计算每个顾客被服务的时间相关信息
    辅助函数，根据给定路径，计算到达该路径上各顾客的时间
    :param customers:各种车型的路线字典，例子：{2: [[1, 4], [5, 6]],3:[8, 12], [13, 16, 17]]}
    :param routes: 各种车型的路线字典，例子：{2: [['o8', 'd2', 'o0', 'd0']], 3:[['o7', 'd0', 'o6', 'd0'], ['o1', 'd1', 'o9', 'd2']}
    :return:
    """
    schedule_dict = defaultdict(OrderSchedule)
    for key, val in customers.items():
        for each_route in val:
            order_schedules = []
            order_n = len(each_route)
            for i in range(order_n):
                order_id = each_route[i]
                origin_id = order_dict[order_id].origin_id
                destination_id = order_dict[order_id].destination_id
                expect_pickup_time = order_dict[order_id].expect_pickup_time
                expect_dropoff_time = order_dict[order_id].expect_dropoff_time
                service_time = order_dict[order_id].service_time

                actual_pickup_time = datetime.datetime(1970, 1, 1, 0, 0, 0)  # 初始化为1970年1月1日的0时0分0秒
                actual_dropoff_time = datetime.datetime(1970, 1, 1, 0, 0, 0)  # 同上
                transit_time = datetime.timedelta(seconds=0)  # 使用datetime.timedelta来表示持续时间，初始为0秒
                leave_time = datetime.datetime(1970, 1, 1, 0, 0, 0)  # 同上

                # 使用不同的参数来初始化datetime.datetime对象，根据你的需求更改日期和时间

                if i == 0:
                    # 当订单是第一个时，实际服务时间为期望服务时间
                    actual_pickup_time = expect_pickup_time
                    # 离开时间=实际服务时间+服务时长
                    leave_time = actual_pickup_time + datetime.timedelta(seconds=service_time)
                    actual_dropoff_time = leave_time + datetime \
                        .timedelta(seconds=cost_dict[(origin_id, destination_id)].duration)
                    if i + 1 < order_n:
                        # 车辆路径上不止1个顾客，上一个顾客到下一个顾客有在途时间
                        # 在途时间=cost[(上个顾客的终点,下个顾客的起点)].duration。
                        # 下个customer（order）的id
                        sub_order_id = each_route[i + 1]
                        transit_time = cost_dict[(order_dict[order_id].destination_id,
                                                  order_dict[sub_order_id].origin_id)].duration
                elif i >= 1:
                    # 不是第一个，有前一个顾客。实际服务时间=max(前一个顾客离开时间+在途时间,期望服务时间)
                    # pre_order_id: 前一个顾客id
                    pre_order_id = each_route[i - 1]
                    actual_pickup_time = max(schedule_dict[pre_order_id].leave_time + datetime.timedelta(
                        seconds=schedule_dict[pre_order_id].transit_time),expect_pickup_time)

                    if i + 1 < order_n:
                        # 不是最后一个顾客
                        sub_order_id = each_route[i + 1]
                        transit_time = cost_dict[(order_dict[order_id].destination_id,
                                                  order_dict[sub_order_id].origin_id)].duration
                # 将订单服务时间等信息存到schedule_dict里面
                schedule_dict[order_id] = OrderSchedule(order_id, expect_pickup_time, expect_dropoff_time,
                                                        actual_pickup_time, actual_dropoff_time,
                                                        service_time, leave_time, transit_time)

    return schedule_dict



def timePenalty(ind, routes):
    '''辅助函数，对不能按服务时间到达顾客的情况进行惩罚'''
    timeArrangement = timeTable(routes)  # 对给定路线，计算到达每个客户的时间
    # 索引给定的最迟到达时间
    desiredTime = [dataDict['Timewindow'][ind[i]][1] for i in range(len(ind))]
    # 如果最迟到达时间大于实际到达客户的时间，则延迟为0，否则延迟设为实际到达时间与最迟到达时间之差
    timeDelay = [max(timeArrangement[i] - desiredTime[i], 0) for i in range(len(ind))]
    return np.sum(timeDelay) / len(timeDelay)



def evaluate(ind, c1=10.0, c2=500.0):
    """评价函数，返回解码后路径的总长度，c1, c2分别为车辆超载与不能服从给定时间窗口的惩罚系数"""
    routes = decodeInd(ind)[1]  # 将个体解码为路线
    total_cost = cal_total_route_cost(routes)
    return (total_cost + c1 * loadPenalty(routes) + c2 * timePenalty(ind, routes)),


# -----------------------------------
# 交叉操作
def genChild(ind1, ind2, nTrail=5):
    '''参考《基于电动汽车的带时间窗的路径优化问题研究》中给出的交叉操作，生成一个子代'''
    # 在ind1中随机选择一段子路径subroute1，将其前置
    routes1 = decodeInd(ind1)  # 将ind1解码成路径
    numSubroute1 = len(routes1)  # 子路径数量
    subroute1 = routes1[np.random.randint(0, numSubroute1)]
    # 将subroute1中没有出现的顾客按照其在ind2中的顺序排列成一个序列
    unvisited = set(ind1) - set(subroute1)  # 在subroute1中没有出现访问的顾客
    unvisitedPerm = [digit for digit in ind2 if digit in unvisited]  # 按照在ind2中的顺序排列
    # 多次重复随机打断，选取适应度最好的个体
    bestRoute = None  # 容器
    bestFit = np.inf
    for _ in range(nTrail):
        # 将该序列随机打断为numSubroute1-1条子路径
        breakPos = [0] + random.sample(range(1, len(unvisitedPerm)), numSubroute1 - 2)  # 产生numSubroute1-2个断点
        breakPos.sort()
        breakSubroute = []
        for i, j in zip(breakPos[0::], breakPos[1::]):
            breakSubroute.append([0] + unvisitedPerm[i:j] + [0])
        breakSubroute.append([0] + unvisitedPerm[j:] + [0])
        # 更新适应度最佳的打断方式
        # 将先前取出的subroute1添加入打断结果，得到完整的配送方案
        breakSubroute.append(subroute1)
        # 评价生成的子路径
        routesFit = calRouteLen(breakSubroute) + loadPenalty(breakSubroute)
        if routesFit < bestFit:
            bestRoute = breakSubroute
            bestFit = routesFit
    # 将得到的适应度最佳路径bestRoute合并为一个染色体
    child = []
    for eachRoute in bestRoute:
        child += eachRoute[:-1]
    return child + [0]


def crossover(ind1, ind2):
    """交叉操作"""
    ind1[:], ind2[:] = genChild(ind1, ind2), genChild(ind2, ind1)
    return ind1, ind2


# -----------------------------------
# 突变操作
def opt(route, dataDict=dataDict, k=2, c1=1.0, c2=500.0):
    # 用2-opt算法优化路径
    # 输入：
    # route -- sequence，记录路径
    # k -- k-opt，这里用2opt
    # c1, c2 -- 寻求最短路径长度和满足时间窗口的相对重要程度
    # 输出： 优化后的路径optimizedRoute及其路径长度
    nCities = len(route)  # 城市数
    optimizedRoute = route  # 最优路径
    desiredTime = [dataDict['Timewindow'][route[i]][1] for i in range(len(route))]
    serviceTime = calcRouteServiceTime(route)
    timewindowCost = [max(serviceTime[i] - desiredTime[1:-1][i], 0) for i in range(len(serviceTime))]
    timewindowCost = np.sum(timewindowCost) / len(timewindowCost)
    minCost = c1 * calRouteLen([route]) + c2 * timewindowCost  # 最优路径代价
    for i in range(1, nCities - 2):
        for j in range(i + k, nCities):
            if j - i == 1:
                continue
            reversedRoute = route[:i] + route[i:j][::-1] + route[j:]  # 翻转后的路径
            # 代价函数中需要同时兼顾到达时间和路径长度
            desiredTime = [dataDict['Timewindow'][reversedRoute[i]][1] for i in range(len(reversedRoute))]
            serviceTime = calcRouteServiceTime(reversedRoute)
            timewindowCost = [max(serviceTime[i] - desiredTime[1:-1][i], 0) for i in range(len(serviceTime))]
            timewindowCost = np.sum(timewindowCost) / len(timewindowCost)
            reversedRouteCost = c1 * calRouteLen([reversedRoute]) + c2 * timewindowCost
            # 如果翻转后路径更优，则更新最优解
            if reversedRouteCost < minCost:
                minCost = reversedRouteCost
                optimizedRoute = reversedRoute
    return optimizedRoute


def mutate(ind):
    '''用2-opt算法对各条子路径进行局部优化'''
    routes = decodeInd(ind)
    optimizedAssembly = []
    for eachRoute in routes:
        optimizedRoute = opt(eachRoute)
        optimizedAssembly.append(optimizedRoute)
    # 将路径重新组装为染色体
    child = []
    for eachRoute in optimizedAssembly:
        child += eachRoute[:-1]
    ind[:] = child + [0]
    return ind,


# -----------------------------------
# 注册遗传算法操作
toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.Individual, genInd)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', crossover)
toolbox.register('mutate', mutate)

# 生成初始族群
toolbox.popSize = 100
pop = toolbox.population(toolbox.popSize)

# 记录迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)
stats.register('std', np.std)
hallOfFame = tools.HallOfFame(maxsize=1)

## 遗传算法参数
toolbox.ngen = 400
toolbox.cxpb = 0.8
toolbox.mutpb = 0.1

## 遗传算法主程序
pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=toolbox.popSize,
                                         lambda_=toolbox.popSize, cxpb=toolbox.cxpb, mutpb=toolbox.mutpb,
                                         ngen=toolbox.ngen, stats=stats, halloffame=hallOfFame, verbose=True)


def calLoad(routes):
    loads = []
    for eachRoute in routes:
        routeLoad = np.sum([dataDict['Demand'][i] for i in eachRoute])
        loads.append(routeLoad)
    return loads


bestInd = hallOfFame.items[0]
distributionPlan = decodeInd(bestInd)
bestFit = bestInd.fitness.values
print('最佳运输计划为：')
pprint(distributionPlan)
print('总运输距离为：')
print(evaluate(bestInd, c1=0, c2=0))
print('各辆车上负载为：')
print(calLoad(distributionPlan))

timeArrangement = timeTable(distributionPlan)  # 对给定路线，计算到达每个客户的时间
# 索引给定的最迟到达时间
desiredTime = [dataDict['Timewindow'][bestInd[i]][1] for i in range(len(bestInd))]
# 如果最迟到达时间大于实际到达客户的时间，则延迟为0，否则延迟设为实际到达时间与最迟到达时间之差
timeDelay = [max(timeArrangement[i] - desiredTime[i], 0) for i in range(len(bestInd))]
print('到达各客户的延迟为：')
print(timeDelay)

# 画出迭代图
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()

## 计算结果如下：
##最佳运输计划为：
# [[0, 1, 2, 3, 4, 6, 0],
# [0, 9, 0],
# [0, 14, 29, 17, 30, 26, 18, 23, 21, 0],
# [0, 8, 25, 15, 16, 0],
# [0, 10, 11, 24, 12, 0],
# [0, 5, 7, 0],
# [0, 28, 27, 20, 19, 22, 13, 0]]
# 总运输距离为：
# (278.62210617851554,)
# 各辆车上负载为：
# [200, 30, 150, 131, 120, 110, 160]
# 到达各客户的延迟为：
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
