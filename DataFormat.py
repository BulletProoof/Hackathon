
INF = 0x3f3f3f


class Order:
    def __init__(self, id, origin_id, destination_id, expect_pickup_time,
                 expect_dropoff_time, volume, service_time, max_waiting_time):
        self.id = id
        self.origin_id = origin_id
        self.destination_id = destination_id
        self.expect_pickup_time = expect_pickup_time
        self.expect_dropoff_time = expect_dropoff_time
        self.volume = volume
        self.service_time = service_time
        self.max_waiting_time = max_waiting_time

    def __str__(self):
        return f"Order(id={self.id}, origin_id={self.origin_id}, destination_id={self.destination_id}, " \
               f"expect_pickup_time={self.expect_pickup_time}, expect_dropoff_time={self.expect_dropoff_time}, " \
               f"volume={self.volume}, service_time={self.service_time}, max_waiting_time={self.max_waiting_time})"


class Node:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __str__(self):
        return f"Node(id={self.id}, x={self.x}, y={self.y})"


class Cost:
    def __init__(self, origin_id, destination_id, duration=0, cost=0):
        self.origin_id = origin_id
        self.destination_id = destination_id
        self.duration = duration
        self.cost = cost

    def __str__(self):
        return f"Cost(origin_id={self.origin_id}, destination_id={self.destination_id}, " \
               f"duration={self.duration}, cost={self.cost})"


class Vehicle:
    def __init__(self, type, lower, upper, cost):
        self.type = type
        self.lower = lower
        self.upper = upper
        self.price = cost

    def __str__(self):
        return f"Vehicle(type={self.type}, lower={self.lower}, " \
               f"upper={self.upper}, price={self.price})"


class OrderSchedule:
    """
    定义每个顾客的：
    期望服务时间、期望卸货时间、实际服务时间、实际卸货时间、服务时长、离开时间（=实际服务时间+服务时长）
    在途时间（若是最后一个顾客，时间为0）
    下一个顾客： 实际服务时间 = 上一顾客的离开时间 + 在途时间
    """

    def __init__(self, order_id, expect_pickup_time, expect_dropoff_time,
                 actual_pickup_time, actual_dropoff_time, service_time, leave_time, transit_time=0):
        self.order_id = order_id
        self.expect_pickup_time = expect_pickup_time
        self.expect_dropoff_time = expect_dropoff_time
        self.actual_pickup_time = actual_pickup_time
        self.actual_dropoff_time = actual_dropoff_time
        self.service_time = service_time
        self.leave_time = leave_time
        self.transit_time = transit_time

    def __str__(self):
        return f"OrderSchedual(order_id={self.order_id}, " \
               f"expect_pickup_time={self.expect_pickup_time}, " \
               f"expect_dropoff_time={self.expect_dropoff_time}, " \
               f"actual_pickup_time={self.actual_pickup_time}, " \
               f"actual_dropoff_time={self.actual_dropoff_time}, " \
               f"service_time={self.service_time}, " \
               f"leave_time={self.leave_time})"
