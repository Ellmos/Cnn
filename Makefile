BINARY=cnn
BUILD_DIR=build

all: target

target: $(BUILD_DIR)
	cmake --build $(BUILD_DIR)
	mv $(BUILD_DIR)/$(BINARY) .

$(BUILD_DIR):
	cmake -D CMAKE_BUILD_TYPE=Debug -B $(BUILD_DIR)


run: target
	./$(BINARY)

clean:
	$(RM) $(BINARY)
	$(MAKE) -C $(BUILD_DIR) clean

dist-clean:
	$(RM) $(BINARY)
	rm -rf $(BUILD_DIR)

.PHONY: all clean run dist-clean $(BINARY) $(BUILD_DIR)
