include ../flags.mk

MODULE = layer
SRC = convolve_layer.cc \
	  dense_layer.cc \
	  pooling_layer.cc \
	  activation_layer.cc \
	  flatten_layer.cc

OBJ = $(SRC:.cc=.o)
LIB = lib$(MODULE).a


all: $(MODULE)

$(MODULE): $(LIB)($(OBJ))

include ../libs.mk

clean:
	$(RM) $(OBJ) $(LIB) $(addprefix $(SRC_DIR), $(LIBS))

.PHONY: all clean $(MODULE)
