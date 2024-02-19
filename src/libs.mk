SRC_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

matrix/libmatrix.a:
	$(MAKE) -C $(SRC_DIR)matrix

neural/libneural.a:
	$(MAKE) -C $(SRC_DIR)neural
