# PLEASE NOTE:
# 1. THIS CODE CONTAINS PIN MAPPINS WHICH ARE ADJUSTED TO THE JETSON XAVIER NX215B.
# 2. EITHER THE FUNCTIONS SHOULD BE CALLED WITH ROOT PRIVILEGES, OR /sys/class/gpio SHOULD HAVE 777 privileges
#
##################################################
# USAGE EXAMPLE:
# from gpio import Gpio
# import time
# from threading import Thread
# alert_gpio = Gpio(17, is_output=True, init_value=0) #init pin 17 as output
# pulse_thread = alert_gpio.trigger_pulse(17, 2000, 1)
# # optional - wait for pulse to finish
# if (pulse_thread is not None):
#     pulse_thread.join()
# #led blink every 0.1 seconds:
# while(1):
#     alert_gpio.set_pin_value(17, 1)
#     time.sleep(0.1)
#     alert_gpio.set_pin_value(17, 0)
#     time.sleep(0.1)

from os import path
from threading import Thread
from threading import Lock
import time
import enum
from .analyzer_general import logger
from general import apply_from_dict
from argparse import Namespace

SYS_GPIO = "/sys/class/gpio"
SYS_GPIO_EXPORT = "/sys/class/gpio/export"
SYS_GPIO_UNEXPORT = "/sys/class/gpio/unexport"

XAVIER_NX215B_INF = { #panel pin -> gpio pin id
    17: 445,
    18: 446,
    19: 447,
    20: 448,
}

class e_AlertActionGpioType(enum.IntEnum):
    Pulse = (0,)
    Level = (1,)

class e_AlertActionGpioActive(enum.IntEnum):
    Low = (0,)
    High = (1,)

class Gpio:
    OUT = "out"
    IN = "in"
    instances = {pin: None for pin in XAVIER_NX215B_INF}
    lock = Lock()

    def __init__(self, panel_pin=None, is_output=True, set_init=True, init_value=0):
        if (panel_pin is not None):
            if (panel_pin not in XAVIER_NX215B_INF):
                logger.error(f'Pin {panel_pin} not available')
                return None
            self.lock.acquire()
            if (self.instances[panel_pin] is None):
                self.instances[panel_pin] = _GpioPin(panel_pin, is_output, set_init=set_init, init_value=init_value)
            self.lock.release()

    ##################################################
    # This function returns a list of available pins
    ##################################################
    def get_avail_pins(self):
        return XAVIER_NX215B_INF.keys()

    ##################################################
    # This function sets the value of the pin for a given duration
    # After the duration ends, the pin is set to the opposite of pulse_value
    # pulse_value is either 1 or 0
    # This function is non-blocking, but only one pulse can be generated at a time
    ##################################################
    def trigger_pulse(self, panel_pin, duration_ms, pulse_value=1):
        if not (self.is_pin_valid(panel_pin)):
            logger.error(f'Pin {panel_pin} not available')
        return self.instances[panel_pin].trigger_pulse(duration_ms, pulse_value)

    def is_pin_valid(self, panel_pin):
        if (panel_pin not in self.instances):
            return False
        if (self.instances[panel_pin] is None):
            logger.error(f'Pin {panel_pin} was not initialized')
            return False
        return True

    ##################################################
    # This function returns the value of the pin
    # On success, returned value is 0 or 1
    # On failure, returned value is None
    ##################################################
    def get_pin_value(self, panel_pin):
        if not (self.is_pin_valid(panel_pin)):
            return None
        return self.instances[panel_pin].get_pin_value()

    ##################################################
    # This function sets the value of the pin
    # The value is either 0 or 1
    # This function can be applied only on ouput pins
    ##################################################
    def set_pin_value(self, panel_pin, value):
        if not (self.is_pin_valid(panel_pin)):
            return False
        return self.instances[panel_pin].set_output_value(value)

    ##################################################
    # This function returns the direction of the pin
    # On success, returned value is ("out" or "in")
    # On failure, returned value is None
    ##################################################
    def get_pin_direction(self, panel_pin):
        if not (self.is_pin_valid(panel_pin)):
            return False
        return self.instances[panel_pin].get_pin_direction()

    ##################################################
    # The dicrection is either "out" or "in"
    ##################################################
    def set_pin_direction(self, panel_pin, direction):
        if (direction not in [Gpio.OUT, Gpio.IN]):
            logger.error(f'invalid direction {direction} for pin {panel_pin}')
            return False
        if not (self.is_pin_valid(panel_pin)):
            return None
        return self.instances[panel_pin].set_pin_direction(direction)

class _GpioPin:
    lock = None
    pin_id = None
    panel_pin_id = None
    pin_path = None
    pin_value_path = None
    pin_direction_path = None

    def __init__(self, panel_pin, is_output=True, set_init=True, init_value=0):
        self.pin_id = XAVIER_NX215B_INF[panel_pin]
        self.panel_pin_id = panel_pin
        self.lock = Lock()
        self.pin_path = SYS_GPIO + "/gpio" + str(self.pin_id)
        self.pin_value_path = self.pin_path + "/value"
        self.pin_direction_path = self.pin_path + "/direction"
        self.enable_pin()
        self.set_pin_direction(Gpio.OUT if is_output else Gpio.IN)
        if (is_output and set_init and init_value in [0, 1]):
            self.set_output_value(init_value)

    def enable_pin(self):
        if self.is_pin_enabled():
            return True
        else:
            try:
                logger.info(f'enabling {self.pin_id}: {self.panel_pin_id}')
                with open(SYS_GPIO_EXPORT, 'w') as f:
                    f.write(str(self.pin_id))
                    return True
            except:
                return False

    def is_pin_enabled(self):
        return path.exists(self.pin_path)

    def trigger_pulse(self, duration_ms, pulse_value=1):
        thread = Thread(target=self.pulse, args=(duration_ms, pulse_value))
        thread.start()
        return thread

    def set_output_value(self, value):
        if (value not in [0, 1]):
            logger.error(f'invalid value {value} for pin {self.panel_pin_id}')
            return False
        try:
            with open(self.pin_value_path, 'w') as f:
                logger.info(f'writing to {self.pin_id}: {self.panel_pin_id} : {value}')
                f.write(str(value))
                return True
        except:
            return False

    def get_pin_value(self):
        try:
            with open(self.pin_value_path, 'r') as f:
                val = f.read(1)
                logger.info(f'read from to {self.pin_id}: {self.panel_pin_id} : {val}')
                return val
        except:
            return None

    def pulse(self, ms, pulse_value):
        locked = self.lock.acquire(blocking=False)
        if (locked):
            self.set_output_value(pulse_value)
            sleep_sec = ms/1000
            time.sleep(sleep_sec)
            self.set_output_value(1 - pulse_value)
            self.lock.release()
        else:
            logger.error(f'failed to generate pulse on pin {self.panel_pin_id}: (lock was taken)')

    def get_pin_direction(self):
        try:
            with open(self.pin_direction_path, 'r') as f:
                val = f.read(3).strip()
                logger.info(f'direction of {self.pin_id}: {self.panel_pin_id} : {val}')
                return val
        except:
            return None

    def set_pin_direction(self, direction):
        try:
            with open(self.pin_direction_path, 'w') as f:
                logger.info(f'setting direction to {self.pin_id}: {self.panel_pin_id} : {direction}')
                f.write(direction)
                return True
        except:
            return False

def sendGPIO(gpio_action):
    if gpio_action.type == e_AlertActionGpioType.Level.value:
        gpio_action.gpio.set_pin_value(gpio_action.id,gpio_action.active)
    else:
        gpio_action.gpio.trigger_pulse(gpio_action.id,gpio_action.pw,gpio_action.active)

def gpio_builder(value):
    gpio_id = apply_from_dict("gpio", value, None)
    if not(gpio_id in XAVIER_NX215B_INF):
        return None
    gpio_level = apply_from_dict("level", value, 0)
    gpio_pw = apply_from_dict("duration", value, 1)
    gpio_set_init = True

    if gpio_level:
        gpio_init = 1
        gpio_active = 0
    else:
        gpio_init = 0
        gpio_active = 1

    # set up the GPIO
    gpio = Gpio(gpio_id, set_init=gpio_set_init, init_value=gpio_init)
    return Namespace(id=gpio_id, type=e_AlertActionGpioType.Pulse.value, active=gpio_active, pw=gpio_pw, gpio=gpio)
