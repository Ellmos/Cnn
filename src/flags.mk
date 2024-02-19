SRC_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

CC = g++
CPPFLAGS = -I$(SRC_DIR)
CXXFLAGS = -Wall -Wextra -Werror -Wvla -pedantic

CXXFLAGS += -g -fsanitize=address
LDFLAGS = -fsanitize=address
LDLIBS = -lsasan

AR = ar
ARFLAGS = rcvs
