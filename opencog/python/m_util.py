##
# @file m_util.py
# @brief developing python library
# @author Dingjie.Wang
# @version 1.0
# @date 2012-08-04


import re
import inspect

def replace_with_dict(s, d):
    '''replace key words in a string with dictionary '''
    try:
        pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')
        return pattern.sub(lambda x: d[x.group()], s)
    except Exception:
        return s

def format_log(offset, dsp_caller = True, *args):
    '''  '''
    caller = "" 
    if dsp_caller:
        stack = inspect.stack()
        caller = " -- %s %s" % (stack[1][2], stack[1][3])

    out =  ' ' * offset +  ' '.join(map(str, args)) + caller
    return out

class Logger(object):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    def __init__(self, f = 'opencog-python.log'):
        try:
            self._file = open(f,'w')
        except IOError:
            print " error: can't open logging file %s " % f
            #raise IOError
        self._filename = f
        self.to_stdout = True
        self._levels = set()
        #
        self.to_file = True
        self.ident = 0
        self.add_level(Logger.ERROR)
    
    def debug(self,msg):

        try:
            if self.to_file and Logger.DEBUG in self._levels:
                print >>self._file, "[DEBUG]:"  + str(msg)
            if self.to_stdout and Logger.DEBUG in self._levels:
                print "[DEBUG]:" +  str(msg)
                #pprint("[DEBUG]:"  + str(msg))
        except IOError:
            print " error: can't write logging file %s " % self._filename

    def info(self, msg):
        try:
            if self.to_file and Logger.INFO in self._levels:
                print >>self._file, "[INFO]:"  + str(msg)
            if self.to_stdout and Logger.INFO in self._levels:
                print "[INFO]:" +  str(msg)
        except IOError:
            print " error: can't write logging file %s " % self._filename
            

    def warning(self,msg):
        try:
            if self.to_file and Logger.WARNING in self._levels:
                print >>self._file, "[WARNING]:"  + str(msg)
            if self.to_stdout and Logger.WARNING in self._levels:
                print "[WARNING]:" +  str(msg)
        except IOError:
            print " error: can't write logging file %s " % self._filename

    def error(self, msg):
        try:
            if self.to_file and Logger.ERROR in self._levels:
                print >>self._file, "[ERROR]:"  + str(msg)
            if self.to_stdout and Logger.ERROR in self._levels:
                print "[ERROR]:" +  str(msg)
        except IOError:
            print " error: can't write logging file %s " % self._filename

    def flush(self):
        self._file.flush()
    
    def use_stdout(self, use):
        self.to_stdout = use

    #def setLevel(self, level):
        #self._levels.append(level)
    def add_level(self, level):
        self._levels.add(level)
        '''docstring for add_level''' 
log = Logger()
#log.add_level(Logger.ERROR)
__all__ = ["log", "Logger", "replace_with_dict"]
