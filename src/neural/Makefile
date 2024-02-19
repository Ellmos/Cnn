include ../flags.mk

MODULE = neural
OBJ = $(MODULE).o
LIB = lib$(MODULE).a

all: $(MODULE)

$(MODULE): $(LIB)($(OBJ))

include ../libs.mk

clean:
	$(RM) $(OBJ) $(LIB) $(addprefix $(SRC_DIR), $(LIBS))

.PHONY: all clean $(MODULE)
