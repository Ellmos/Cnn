export CC = g++
export CPPFLAGS = -Isrc
export CXXFLAGS = -Wall -Wextra -Werror -pedantic -std=c++20 -Wold-style-cast

BINARY = cnn

all: $(BINARY)

$(BINARY):
	$(MAKE) -C src
	mv src/$(BINARY) .

run: $(BINARY)
	@echo "----------------------------------------"
	@./$(BINARY)


clean:
	$(MAKE) -C src clean
	$(RM) $(BINARY)

.PHONY: all clean run $(BINARY)
