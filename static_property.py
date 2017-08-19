# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 13:04:00 2017

@author: maximilian_fritz
"""

class static_property(object):
    """ A data descriptor. """
    def __init__(self, func):
        self.func = func
        self.var = '_' + func.__name__
    
    def __get__(self, instance, cls):
        if not hasattr(instance, self.var):
            setattr(instance, self.var, self.func(instance))
        return getattr(instance, self.var)
        

if __name__ == '__main__':
    class Foo(object):
    
        def __init__(self):
            pass
        
        @static_property
        def foo_property(self):
            return True
        
        
    f = Foo()
    print(f.foo_property)