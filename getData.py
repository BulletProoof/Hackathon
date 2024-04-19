import csv
from collections import defaultdict
import re
from datetime import datetime
import time

from DataFormat import *

def get_data(file_path=None):
    start_time = time.time()

    # 使用DataDefine定义后
    # 创建Cost字典，存放数据；key：（origin_id,destination_id）val:Cost对象
    cost_dict = defaultdict(Cost)

    # 创建车辆字典,key:type, val:Vehicle
    vehicle_dict = defaultdict(Vehicle)

    # 创建点的字典，key:id,key:Node
    node_dict = defaultdict(Node)

    # 创建订单字典，key:order_id,val:Order
    order_dict = defaultdict(Order)

    """
    # 创建距离和配送时间字典，存放数据
    dist_time_dict = defaultdict(lambda: (0x3f3f3f, 0x3f3f3f))

    # 存放点的数据
    node_dict = defaultdict(lambda: (0, 0))

    # 存放车辆的数据
    vehicle_dict = defaultdict(lambda: (0, 0, 0))

    # 存放订单的数据:order_id,origin_id,destination_id
    # expect_pickup_time,expect_dropoff_time,
    # volume,service_time,max_waiting_time
    order_dict = defaultdict(lambda: (0, 0, 0, 0, 0, 0, 0))
    """

    with open('data/transport_cost.csv', 'r') as file:

        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            origin_id = str(row[0])
            destination_id = str(row[1])
            duration = int(row[2])
            cost = float(row[3])
            cost_dict[(origin_id, destination_id)] = Cost(origin_id, destination_id, duration, cost)
            # dist_time_dict[(origin_id, destination_id)] = (duration, cost)

    with open('data/coordinate_data.csv', 'r') as file:

        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            node_id = str(row[0])
            x_coordinate = float(row[1])
            y_coordinate = float(row[2])
            node_dict[node_id] = Node(node_id,x_coordinate,y_coordinate)
            # node_dict[node_id] = (x_coordinate, y_coordinate)

    with open('data/vehicle_data.csv', 'r') as file:

        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            type_id = int(row[0])
            volume_upper_lower = str(row[1])
            cost = float(row[2])

            # 解析volume_upper_lower->lower,upper
            lower, upper = re.findall(r'\d+\.\d+', volume_upper_lower)
            lower = float(lower)
            upper = float(upper)
            vehicle_dict[type_id] = Vehicle(type_id, lower, upper, cost)
            # vehicle_dict[vehicle_type_id] = (lower, upper, cost)

    with open('data/order_data.csv', 'r') as file:

        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Skip the header row
        next(csv_reader)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            order_id = int(row[0])
            origin_id = str(row[1])
            destination_id = str(row[2])

            # 解析期待服务和期待送达时间格式
            expect_pickup_time = row[3]
            expect_dropoff_time = row[4]

            date_format = "%d/%m/%Y %H:%M"
            expect_pickup_time = datetime.strptime(expect_pickup_time, date_format)
            expect_dropoff_time = datetime.strptime(expect_dropoff_time, date_format)

            volume = float(row[5])
            service_time = int(row[6])
            max_waiting_time = int(row[6])

            order_dict[order_id] = Order(order_id, origin_id, destination_id, expect_pickup_time,
                                         expect_dropoff_time, volume, service_time,max_waiting_time)
            # order_dict[order_id] = (origin_id, destination_id,
            #                                expect_pickup_time, expect_dropoff_time,
            #                                volume, service_time, max_waiting_time)

    end_time = time.time()
    run_time = end_time - start_time
    print("读入数据花费时间为：", run_time, "秒")
    return node_dict, vehicle_dict, cost_dict, order_dict
    # return node_dict, vehicle_dict, dist_time_dict, order_dict


