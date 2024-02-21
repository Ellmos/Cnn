include ../flags.mk

MODULE = layer
SRC = convolve_layer.cpp \
	  dense_layer.cpp \
	  pooling_layer.cpp \
	  activation_layer.cpp \
	  flatten_layer.cpp

OBJ = $(SRC:.cpp=.o)
LIB = lib$(MODULE).a


all: $(MODULE)

$(MODULE): $(LIB)($(OBJ))

include ../libs.mk

clean:
	$(RM) $(OBJ) $(LIB) $(addprefix $(SRC_DIR), $(LIBS))

.PHONY: all clean $(MODULE)
