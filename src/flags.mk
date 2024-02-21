SRC_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

CC = g++
CPPFLAGS = -I$(SRC_DIR)
CXXFLAGS = -Wall -Wextra -Werror -Wvla -pedantic -std=c++11

CXXFLAGS += -g -fsanitize=address
LDFLAGS = -fsanitize=address
LDLIBS = -lsasan

AR = ar
ARFLAGS = rcvs
