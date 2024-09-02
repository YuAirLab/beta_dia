import datetime
import logging
import time

class MyFormatter(logging.Formatter):
    def format(self, record):
        ms = record.relativeCreated
        delta_time = datetime.timedelta(milliseconds=ms)
        clock_time = datetime.datetime(1, 1, 1) + delta_time
        clock_string = clock_time.strftime("%H:%M:%S")
        record.adjustedTime = clock_string
        return super(MyFormatter, self).format(record)


class Logger():
    # class variables
    logger = logging.getLogger('Beta-DIA')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False # no forward transfer

    @classmethod
    def set_logger(cls, dir_out, is_time_name=True):
        logging._startTime = time.time() # reset relative time

        # fh
        logtime = time.strftime("%Y_%m_%d_%H_%M")
        if is_time_name:
            fname = logtime + '.log.txt'
        else:
            fname = 'report_beta.log.txt'
        fh = logging.FileHandler(dir_out / fname, mode='w')
        fh.setLevel(logging.INFO)

        # ch
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # format to handler
        formatter = MyFormatter(
            '%(adjustedTime)s: %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # handler binding to logger
        for handler in cls.logger.handlers:
            if type(handler) is logging.FileHandler:
                cls.logger.removeHandler(handler)
        cls.logger.addHandler(fh)

        for handler in cls.logger.handlers:
            if type(handler) is logging.StreamHandler:
                cls.logger.removeHandler(handler)
        cls.logger.addHandler(ch)

    @classmethod
    def get_logger(cls):
        return cls.logger
