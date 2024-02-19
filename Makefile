export CC = g++
export CPPFLAGS = -Isrc
export CXXFLAGS = -Werror -Wall -Wextra -Wvla -pedantic

BINARY = cnn

all: $(BINARY)

$(BINARY):
	$(MAKE) -C src
	mv src/$(BINARY) .

run: $(BINARY)
	@./$(BINARY)


clean:
	$(MAKE) -C src clean
	$(RM) $(BINARY)

.PHONY: all clean run $(BINARY)
