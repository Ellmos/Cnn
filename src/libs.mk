SRC_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

# matrix/libmatrix.a:
# 	$(MAKE) -C $(SRC_DIR)matrix

layer/liblayer.a:
	$(MAKE) -C $(SRC_DIR)layer

neural/libneural.a:
	$(MAKE) -C $(SRC_DIR)neural
