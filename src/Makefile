include flags.mk

BINARY=cnn
OBJ = cnn.o

LIBS = layer/liblayer.a
	   # matrix/libmatrix.a
	   # neural/libneural.a


all: $(BINARY)

$(BINARY): $(OBJ) $(LIBS)
	$(CC) $(LDFLAGS) -o $@ $(OBJ) -Wl,--start-group $(LIBS) -Wl,--end-group

include libs.mk

clean: 
	$(MAKE) -C layer clean 
	$(MAKE) -C neural clean 
	$(RM) $(OBJ)


.PHONY: all clean $(MODULES) $(BINARY)
