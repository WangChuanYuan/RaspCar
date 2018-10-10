#!/bin/bash
g++ $1 -o car `pkg-config --cflags --libs opencv` -L. -lwiringPi -lGPIO -lpthread