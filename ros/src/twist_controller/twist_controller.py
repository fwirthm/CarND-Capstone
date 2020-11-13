from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy 

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self,\
                 wheel_base, steer_ratio,\
                 min_speed, max_lat_accel,\
                 max_steer_angle, vehicle_mass,\
                 wheel_radius, decel_limit):
        # initialize yaw controller object
        self.yaw_controller = YawController(wheel_base, steer_ratio,\
                                            min_speed, max_lat_accel,\
                                            max_steer_angle)
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0.
        mx = 0.2
        
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        tau = 0.2
        ts = 0.02
        
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.last_time = rospy.get_time()
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.decel_limit = decel_limit
     

    def controll(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # use controllers to calculate throttle, brake and steer commands

        if not dbw_enabled:
            self.throttle_controller.reset()
            return (0., 0., 0.)
        
        #lowpass filter current velocity - otherwise teh steering controller won't work
        current_vel = self.vel_lpf.filt(current_vel)
        
        #use yaw controller to get steering commands
        steer = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        vel_err = linear_vel - current_vel
        self.last_vel = current_vel
        
        #calculate the time since the last control action
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_err, sample_time)
        brake = 0.
        if linear_vel == 0. and current_vel<0.1:
            throttle = 0.
            brake = 700.
        elif throttle <0.1 and vel_err<0.:
            throttle = 0.
            decel =  max(vel_err, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
            
        
        # Return throttle, brake, steer
        return (throttle, brake, steer)
