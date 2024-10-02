#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Internal Logger configuration
"""

import logging
import sys

# LEVELS
# logging.DEBUG
# logging.INFO
# logging.WARNING
# logging.ERROR
# logging.CRITICAL
# logging.NOTSET

def get_logger():
    return logging.getLogger()


def set_log_level(log_level):
    logging.getLogger().setLevel(log_level)
